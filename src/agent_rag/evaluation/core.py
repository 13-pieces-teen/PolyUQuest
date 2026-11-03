"""Dataset loading and metrics, independent of databases and model clients."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    short_answers: tuple[str, ...]
    gold_block_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    build_id: str = ""


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return tuple(dict.fromkeys(value))


def load_questions(path: str | Path) -> list[Question]:
    """Load the PolyUQuest benchmark JSON array using its public field names."""
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("Dataset must be a JSON array of question objects")
    questions = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each dataset record must be a question object")
        qid = row.get("id")
        question = row.get("question")
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError(f"Invalid or duplicate question id: {qid!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question text: {qid}")
        answers = _strings(row.get("short_answers"), "short_answers")
        gold = _strings(row.get("gold_block_ids"), "gold_block_ids")
        if not answers and not gold:
            raise ValueError(f"Question {qid} has neither answers nor gold block ids")
        questions.append(
            Question(
                id=qid,
                question=question,
                short_answers=answers,
                gold_block_ids=gold,
                tags=_strings(row.get("tags", []), "tags"),
                build_id=str(row.get("_build_id", "")),
            )
        )
        seen.add(qid)
    if not questions:
        raise ValueError("Dataset is empty")
    return questions


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def answer_matches(context: str, answer: str) -> bool:
    """Match a normalized answer phrase without matching Latin word fragments."""
    context, answer = _normalize(context), _normalize(answer)
    if not answer:
        return False
    # Prevent e.g. year 2026 matching 12026 or short names matching word fragments.
    return bool(re.search(r"(?<![a-z0-9_])" + re.escape(answer) + r"(?![a-z0-9_])", context))


def score_blocks(question: Question, blocks: list[dict], ks: tuple[int, ...]) -> dict:
    """Rank is returned order. Duplicate block IDs cannot inflate recall or rank."""
    unique = []
    seen = set()
    for block in blocks:
        bid = block.get("block_id")
        if bid and bid in seen:
            continue
        if bid:
            seen.add(bid)
        unique.append(block)
    gold = set(question.gold_block_ids)
    scores = {}
    for k in ks:
        selected = unique[:k]
        found = {b.get("block_id") for b in selected} & gold
        texts = [str(b.get("content", "")) for b in selected]
        matches = [any(answer_matches(t, a) for t in texts) for a in question.short_answers]
        scores[f"block_hit@{k}"] = float(bool(found)) if gold else None
        scores[f"block_recall@{k}"] = len(found) / len(gold) if gold else None
        rank = next((i for i, b in enumerate(selected, 1) if b.get("block_id") in gold), 0)
        scores[f"block_mrr@{k}"] = 1 / rank if rank else (0.0 if gold else None)
        scores[f"answer_hit@{k}"] = float(any(matches)) if question.short_answers else None
        answer_rank = next(
            (
                i
                for i, text in enumerate(texts, 1)
                if any(answer_matches(text, a) for a in question.short_answers)
            ),
            0,
        )
        scores[f"answer_mrr@{k}"] = (
            1 / answer_rank if answer_rank else (0.0 if question.short_answers else None)
        )
        scores[f"answer_coverage@{k}"] = sum(matches) / len(matches) if matches else None
    return scores


def aggregate(rows: list[dict]) -> dict:
    values = defaultdict(list)
    for row in rows:
        for key, value in row["metrics"].items():
            if value is not None:
                values[key].append(value)
    return {
        "n_queries": len(rows),
        "n_errors": sum(r["error"] is not None for r in rows),
        "metrics": {key: sum(v) / len(v) for key, v in sorted(values.items())},
        "denominators": {key: len(v) for key, v in sorted(values.items())},
        "avg_latency_seconds": sum(r["latency_seconds"] for r in rows) / len(rows),
    }


def evaluate(
    questions: list[Question],
    retrieve: Callable[[Question], dict[str, Any]],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    if not questions or not ks or any(k < 1 for k in ks):
        raise ValueError("Nonempty questions and positive cutoffs are required")
    rows = []
    for question in questions:
        start = time.perf_counter()
        error = None
        try:
            result = retrieve(question)
            if result.get("error"):
                raise RuntimeError(f"Recorded retrieval failure: {result['error']}")
            blocks = result["blocks"]
            if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
                raise ValueError("Retrieval must return a list of block objects")
        except Exception as exc:
            result, blocks = {}, []
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "id": question.id,
                "tags": question.tags,
                "mode": result.get("mode"),
                "error": error,
                "latency_seconds": time.perf_counter() - start,
                "metrics": score_blocks(question, blocks, ks),
                "blocks": blocks,
                "routing": result.get("routing"),
                "trace": result.get("trace", []),
            }
        )
    tags = sorted({tag for q in questions for tag in q.tags})
    return {
        "summary": aggregate(rows),
        "by_tag": {tag: aggregate([r for r in rows if tag in r["tags"]]) for tag in tags},
        "per_query": rows,
    }
