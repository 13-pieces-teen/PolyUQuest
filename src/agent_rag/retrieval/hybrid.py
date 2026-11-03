"""Hybrid retrieval: run two or three modes in parallel and merge block results.

Used in two scenarios:
- Router returns a low-confidence decision with an ``alt_mode`` (2-mode hybrid)
- Caller forces ``mode="hybrid"`` to run all three modes in parallel

Each sub-mode runs with ``skip_answer=True``, so only the merged-context
answer-generation call hits the LLM — the three (or two) sub-mode LLM
generations are skipped. With LLM cache also enabled, repeat queries hit
SQLite and return in <100ms.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Any, Callable

import structlog
from jinja2 import Template

from agent_rag.config import llm_config, thresholds_config
from agent_rag.llm.client import LLMClient, cost_stage
from agent_rag.retrieval._context import collect_related_links
from agent_rag.retrieval.direct import retrieve_direct
from agent_rag.retrieval.navigation import retrieve_navigation
from agent_rag.retrieval.reasoning import retrieve_reasoning
from agent_rag.storage.neo4j_store import Neo4jStore
from agent_rag.storage.qdrant_store import QdrantStore

logger = structlog.get_logger(__name__)

_hybrid_cfg = thresholds_config.get("retrieval", {}).get("hybrid", {})
MAX_MERGED_BLOCKS = int(_hybrid_cfg.get("max_blocks", 10))

_GENERATION_MODEL = (llm_config.get("generation", {}) or {}).get("model")
_GENERATION_MAX_TOKENS = int((llm_config.get("generation", {}) or {}).get("max_tokens", 2048))

_ANSWER_TMPL = Template(
    (Path(__file__).parent.parent / "llm" / "prompts" / "generate_answer.j2")
    .read_text(encoding="utf-8")
)

_MODE_TO_FN: dict[str, Callable[..., dict[str, Any]]] = {
    "mode_a": retrieve_direct,
    "mode_b": retrieve_navigation,
    "mode_c": retrieve_reasoning,
}


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _normalize_scores(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []
    scores = [float(b.get("score", 0.0) or 0.0) for b in blocks]
    lo = min(scores)
    hi = max(scores)
    span = hi - lo if hi > lo else 1.0
    out = []
    for b, s in zip(blocks, scores):
        nb = dict(b)
        nb["score_raw"] = s
        nb["score"] = (s - lo) / span
        out.append(nb)
    return out


def _merge_blocks(
    blocks_by_mode: dict[str, list[dict[str, Any]]],
    cap: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for mode, blks in blocks_by_mode.items():
        for b in _normalize_scores(blks):
            bid = b.get("block_id", "")
            if not bid:
                continue
            if bid not in merged:
                copy = dict(b)
                copy["from_modes"] = [mode]
                merged[bid] = copy
            else:
                existing = merged[bid]
                if b["score"] > existing["score"]:
                    # keep higher-scored copy's metadata but record both modes
                    existing.update({k: v for k, v in b.items() if k != "from_modes"})
                if mode not in existing["from_modes"]:
                    existing["from_modes"].append(mode)
    out = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return out[:cap]


def retrieve_hybrid(
    query: str,
    query_embedding: list[float],
    neo4j: Neo4jStore,
    qdrant: QdrantStore,
    llm: LLMClient,
    modes: list[str],
    skip_answer: bool = False,
    history: list[Any] | None = None,
    answer_query: str | None = None,
) -> dict[str, Any]:
    """Run ``modes`` concurrently, merge blocks, regenerate a single answer.

    Sub-modes always run with ``skip_answer=True`` to avoid wasted intermediate
    LLM calls (their answers are discarded by the merge step anyway). The
    outer ``skip_answer`` flag controls whether the merged-context answer is
    generated synchronously or returned as a prompt for the SSE streaming
    endpoint.
    """
    valid_modes = [m for m in modes if m in _MODE_TO_FN]
    if len(valid_modes) < 2:
        # degenerate case: fall back to the single-mode path
        mode = valid_modes[0] if valid_modes else "mode_a"
        return _MODE_TO_FN[mode](
            query,
            query_embedding,
            neo4j,
            qdrant,
            llm,
            skip_answer=skip_answer,
            history=history,
            answer_query=answer_query,
        )

    t0 = time.time()
    trace: list[dict[str, Any]] = []
    per_mode_results: dict[str, dict[str, Any]] = {}
    evaluation_errors: dict[str, str] = {}

    # Step 1: parallel retrieval. Sub-modes skip their own answer generation —
    # only the final merged-context call hits the LLM.
    ts = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(valid_modes)) as pool:
        futures = {
            pool.submit(
                copy_context().run,
                _MODE_TO_FN[m],
                query,
                query_embedding,
                neo4j,
                qdrant,
                llm,
                skip_answer=True,
            ): m
            for m in valid_modes
        }
        for fut, m in futures.items():
            try:
                per_mode_results[m] = fut.result()
            except Exception as exc:
                logger.warning("hybrid_submode_failed", mode=m, error=str(exc))
                evaluation_errors[m] = f"{type(exc).__name__}: {exc}"
                per_mode_results[m] = {"answer": "", "blocks": [], "trace": []}

    trace.append({
        "step": "hybrid_parallel",
        "label": f"Parallel retrieval: {', '.join(valid_modes)}",
        "duration_ms": _ms(ts),
        "data": {
            "modes": valid_modes,
            "per_mode_block_counts": {
                m: len(per_mode_results[m].get("blocks", [])) for m in valid_modes
            },
            # embed sub-traces (flattened) so the frontend can still walk every step
            "sub_traces": {m: per_mode_results[m].get("trace", []) for m in valid_modes},
        },
    })

    # Step 2: merge blocks
    ts = time.perf_counter()
    blocks_by_mode = {m: per_mode_results[m].get("blocks", []) for m in valid_modes}
    merged_blocks = _merge_blocks(blocks_by_mode, cap=MAX_MERGED_BLOCKS)
    trace.append({
        "step": "hybrid_merge",
        "label": "Hybrid Block Merge",
        "duration_ms": _ms(ts),
        "data": {
            "merged_blocks": len(merged_blocks),
            "overlap_count": sum(1 for b in merged_blocks if len(b.get("from_modes", [])) > 1),
        },
    })

    # Step 3: final answer generation on merged context
    ts = time.perf_counter()
    related_links = collect_related_links(merged_blocks, neo4j)
    prompt = _ANSWER_TMPL.render(
        query=answer_query or query,
        blocks=merged_blocks,
        related_links=related_links,
        history=history,
    )
    chat_kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": _GENERATION_MAX_TOKENS,
        "use_cache": True,
    }
    if _GENERATION_MODEL:
        chat_kwargs["model"] = _GENERATION_MODEL
    if skip_answer:
        answer = ""
    else:
        with cost_stage("answer"):
            answer = llm.chat(**chat_kwargs)
    trace.append({
        "step": "answer_generation",
        "label": "Hybrid Answer Generation",
        "duration_ms": _ms(ts),
        "data": {
            "prompt_tokens_est": len(prompt.split()),
            "skipped": skip_answer,
        },
    })

    # Aggregate mode-specific enrichment fields from sub-results (best-effort).
    sub_queries = next(
        (per_mode_results[m].get("sub_queries") for m in valid_modes if per_mode_results[m].get("sub_queries")),
        None,
    )
    keywords_extracted = next(
        (per_mode_results[m].get("keywords_extracted") for m in valid_modes if per_mode_results[m].get("keywords_extracted")),
        None,
    )
    entities_expanded = next(
        (per_mode_results[m].get("entities_expanded") for m in valid_modes if per_mode_results[m].get("entities_expanded")),
        None,
    )

    elapsed = time.time() - t0
    return {
        "answer": answer,
        "mode": "hybrid(" + "+".join(m[-1] for m in valid_modes) + ")",
        "hybrid_modes": valid_modes,
        "blocks": merged_blocks,
        "sub_queries": sub_queries,
        "keywords_extracted": keywords_extracted,
        "entities_expanded": entities_expanded,
        "answer_prompt": prompt,
        "elapsed_seconds": round(elapsed, 2),
        "trace": trace,
        "evaluation_errors": evaluation_errors,
    }
