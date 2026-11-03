"""Native retrieval adapter. One variant per process; no serving config is edited."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any

from agent_rag.evaluation.core import Question

VARIANTS = (
    "full",
    "direct_only",
    "no_hybrid",
    "no_entity_layer",
    "no_topic_keywords",
    "first_stage_only",
)


class NativeRetriever:
    def __init__(self, variant: str = "full"):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        # Imports stay lazy: dataset validation and replay need no model services.
        from agent_rag.config import thresholds_config
        from agent_rag.llm.client import LLMClient
        from agent_rag.retrieval import _reranker, hybrid, reasoning, router
        from agent_rag.retrieval._embedding import embed_query
        from agent_rag.retrieval._rewriter import reset_router_confidence, set_router_confidence
        from agent_rag.retrieval.direct import retrieve_direct
        from agent_rag.retrieval.navigation import retrieve_navigation
        from agent_rag.storage.neo4j_store import Neo4jStore
        from agent_rag.storage.qdrant_store import QdrantStore

        self.variant = variant
        self.stack = ExitStack()
        # Restored on close, also on initialization/preflight failures.
        if variant == "first_stage_only":
            previous_mode = _reranker._RERANKER_MODE
            self.stack.callback(setattr, _reranker, "_RERANKER_MODE", previous_mode)
            _reranker._RERANKER_MODE = "first_stage_only"
        self.functions = {
            "mode_a": retrieve_direct,
            "mode_b": retrieve_navigation,
            "mode_c": reasoning.retrieve_reasoning,
        }
        self.hybrid, self.router, self.reasoning = hybrid, router, reasoning
        self.embed = embed_query
        self.set_confidence, self.reset_confidence = set_router_confidence, reset_router_confidence
        self.allowed = {"mode_a", "mode_b"} if variant == "no_entity_layer" else set(self.functions)
        self.threshold = float(
            thresholds_config.get("retrieval", {}).get("router", {}).get("hybrid_threshold", 0.6)
        )
        self.metadata = {
            "variant": variant,
            "retrieval_config": thresholds_config.get("retrieval", {}),
            "reranker_mode": _reranker.mode(),
            "skip_answer": True,
            "cache_policy": "existing disk caches retained; router cache cleared at start",
        }
        from agent_rag.llm.client import snapshot_usage

        self.llm_usage_start = snapshot_usage()
        self.rerank_usage_start = _reranker.snapshot_usage()
        try:
            self.neo4j = Neo4jStore()
            self.stack.callback(self.neo4j.close)
            self.qdrant = QdrantStore()
            self.stack.callback(self.qdrant.close)
            self.llm = LLMClient()
            self.stack.callback(self.llm.close)
            from agent_rag.retrieval._cache import router_cache

            router_cache.clear()
        except Exception:
            self.close()
            raise

    def validate_corpus(self, questions: list[Question], build_id: str) -> dict:
        """Resolve gold ids before spending model calls. Build ID is user-declared."""
        builds = {q.build_id for q in questions if q.build_id}
        if not build_id or (builds and builds != {build_id}):
            raise ValueError(
                f"Corpus build id {build_id!r} does not match dataset {sorted(builds)}"
            )
        gold = {bid for q in questions for bid in q.gold_block_ids}
        missing_graph = gold - self.neo4j.get_all_block_ids()
        # Qdrant IDs are hashed internally; use its existing public adapter.
        found = set()
        ordered_gold = sorted(gold)
        for start in range(0, len(gold), 128):
            batch = ordered_gold[start : start + 128]
            found.update(self.qdrant.retrieve_vectors("blocks", batch))
        missing_vectors = gold - found
        if missing_graph or missing_vectors:
            raise ValueError(
                f"Gold corpus mismatch: {len(missing_graph)} missing graph blocks, "
                f"{len(missing_vectors)} missing vectors; use the matching corpus snapshot"
            )
        return {
            "declared_build_id": build_id,
            "gold_ids_checked": len(gold),
            "all_gold_ids_resolvable": True,
            "snapshot_identity_verified": False,
        }

    def __call__(self, question: Question) -> dict[str, Any]:
        from agent_rag.storage.qdrant_store import (
            reset_forbidden_collections,
            set_forbidden_collections,
        )

        with ExitStack() as scope:
            if self.variant == "direct_only":
                decision = {
                    "mode": "mode_a",
                    "confidence": 1.0,
                    "alt_mode": None,
                    "source": "forced",
                }
            else:
                decision = self.router.route_query(
                    question.question,
                    llm=self.llm,
                    allowed_modes=self.allowed,
                )
            confidence = float(decision.get("confidence", 1.0))
            token = self.set_confidence(confidence)
            scope.callback(self.reset_confidence, token)
            if self.variant == "no_entity_layer":
                token = set_forbidden_collections({"entities", "relations", "topic_keywords"})
                scope.callback(reset_forbidden_collections, token)
            if self.variant == "no_topic_keywords":
                token = self.reasoning.set_topic_keywords_disabled(True)
                scope.callback(self.reasoning.reset_topic_keywords_disabled, token)
            mode = decision["mode"]
            alt = decision.get("alt_mode")
            if mode not in self.allowed or (alt is not None and alt not in self.allowed):
                raise ValueError(f"Router returned a forbidden mode: {decision}")
            embedding = self.embed(question.question)
            if self.variant != "no_hybrid" and alt and mode != alt and confidence < self.threshold:
                result = self.hybrid.retrieve_hybrid(
                    question.question,
                    embedding,
                    self.neo4j,
                    self.qdrant,
                    self.llm,
                    modes=[mode, alt],
                    skip_answer=True,
                )
            else:
                result = self.functions[mode](
                    question.question,
                    embedding,
                    self.neo4j,
                    self.qdrant,
                    self.llm,
                    skip_answer=True,
                )
            # Hybrid catches service failures by design; make them visible to evaluation.
            if result.get("evaluation_errors"):
                raise RuntimeError(str(result["evaluation_errors"]))
            return {**result, "routing": decision}

    def close(self) -> None:
        self.stack.close()

    def usage(self) -> dict:
        from agent_rag.llm.client import diff_usage
        from agent_rag.retrieval import _reranker

        return {
            "llm": diff_usage(self.llm_usage_start),
            "reranker": _reranker.diff_usage(self.rerank_usage_start),
            "note": "No monetary estimate; embedding usage is not included",
        }
