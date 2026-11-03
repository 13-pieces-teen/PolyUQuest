"""Service-free regression checks for data, scoring, native dispatch and ablations."""

import json
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_rag.evaluation.__main__ import load_replay, main
from agent_rag.evaluation.core import (
    Question,
    answer_matches,
    evaluate,
    load_questions,
    score_blocks,
)
from agent_rag.evaluation.runtime import NativeRetriever


def test_public_dataset():
    public = load_questions(Path(__file__).parents[1] / "data/dataset_polyu_300.json")
    assert len(public) == 300
    assert {tag for q in public for tag in q.tags} == {
        "single_hop",
        "multi_page",
        "entity_list",
        "mixed_ab",
        "mixed_ac",
    }
    assert public[0].short_answers


@pytest.mark.parametrize(
    "records",
    [
        {"id": "q", "question": "q", "short_answers": ["a"], "gold_block_ids": ["g"]},
        ["invalid"],
        [{"question": "q", "short_answers": ["a"], "gold_block_ids": ["g"]}],
        [{"id": "q", "question": "q", "gold_block_ids": ["g"]}],
        [{"id": "q", "question": "q", "short_answers": "a", "gold_block_ids": ["g"]}],
        [{"id": "q", "question": "q", "short_answers": ["a"]}],
    ],
)
def test_dataset_requires_public_schema(tmp_path, records):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(records), "utf-8")
    with pytest.raises(ValueError):
        load_questions(path)


def test_metric_denominators_rank_dedup_and_multiple_answers():
    question = Question("q", "q", ("Alice", "Bob"), ("g1", "g2"))
    blocks = [
        {"block_id": "n", "content": "none"},
        {"block_id": "n", "content": "none"},
        {"block_id": "g2", "content": "Alice"},
    ]
    scores = score_blocks(question, blocks, (1, 3))
    assert scores["block_hit@1"] == 0
    assert scores["block_mrr@3"] == 0.5
    assert scores["block_recall@3"] == 0.5
    assert scores["answer_coverage@3"] == 0.5
    assert scores["answer_mrr@3"] == 0.5
    answer_only = Question("unannotated", "q", ("Alice",))
    report = evaluate([question, answer_only], lambda q: {"blocks": blocks})
    assert report["summary"]["denominators"]["block_hit@5"] == 1
    assert report["summary"]["denominators"]["answer_hit@5"] == 2


def test_answer_matching_preserves_phrases_and_numeric_boundaries():
    assert not answer_matches("12026", "2026")
    assert answer_matches("PolyU opened in 2026.", "2026")
    assert answer_matches("这里是香港理工大学的校址", "香港理工大学")
    assert not answer_matches("Bob and Alice", "Alice Bob")


def test_failure_is_counted_as_zero_not_dropped():
    def fail(_):
        raise ConnectionError("offline")

    report = evaluate([Question("q", "q", ("A",), ("g",), ("tag",))], fail)
    assert report["summary"]["n_errors"] == 1
    assert report["by_tag"]["tag"]["metrics"]["block_hit@5"] == 0


def test_replay_cli_and_duplicate_rejection(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text(
        '[{"id":"q","question":"q","short_answers":["answer"],"gold_block_ids":["g"]}]', "utf-8"
    )
    replay = tmp_path / "predictions.jsonl"
    record = {"id": "q", "blocks": [{"block_id": "g", "content": "answer"}]}
    replay.write_text(json.dumps(record), "utf-8")
    output = tmp_path / "output"
    assert (
        main(["--dataset", str(dataset), "--replay", str(replay), "--output-dir", str(output)]) == 0
    )
    assert json.loads((output / "full.json").read_text())["summary"]["metrics"]["block_hit@5"] == 1
    replay.write_text(json.dumps([record, record]), "utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_replay(str(replay), "full")


@pytest.fixture
def native(monkeypatch):
    """Use real dispatch and guard code with inert stores / LLM / embedding."""
    from agent_rag.llm import client
    from agent_rag.retrieval import _embedding
    from agent_rag.storage import neo4j_store, qdrant_store

    store = SimpleNamespace(
        close=lambda: None,
        get_all_block_ids=lambda: {"g"},
        retrieve_vectors=lambda collection, ids: {i: [1.0] for i in ids},
    )
    monkeypatch.setattr(neo4j_store, "Neo4jStore", lambda: store)
    monkeypatch.setattr(qdrant_store, "QdrantStore", lambda: store)
    monkeypatch.setattr(client, "LLMClient", lambda: store)
    monkeypatch.setattr(_embedding, "embed_query", lambda q: [1.0])
    yield


@pytest.mark.parametrize(
    "variant",
    [
        "full",
        "no_entity_layer",
        "no_topic_keywords",
        "no_hybrid",
        "direct_only",
        "first_stage_only",
    ],
)
def test_native_variants_dispatch_and_restore(native, monkeypatch, variant):
    from agent_rag.retrieval import _reranker, reasoning
    from agent_rag.retrieval._rewriter import _gating_ctx
    from agent_rag.storage.qdrant_store import ForbiddenCollectionError, _check_forbidden

    previous_mode = _reranker.mode()
    runtime = NativeRetriever(variant)
    calls = []

    def retrieve(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["skip_answer"] is True
        assert reasoning._topic_keywords_disabled.get() == (variant == "no_topic_keywords")
        if variant == "no_entity_layer":
            with pytest.raises(ForbiddenCollectionError):
                _check_forbidden("entities")
        assert _gating_ctx.get()[0] == 0.5 or variant == "direct_only"
        return {"blocks": [{"block_id": "g", "content": "answer"}]}

    monkeypatch.setattr(
        runtime.router,
        "route_query",
        lambda *a, **kw: {
            "mode": "mode_a",
            "confidence": 0.5,
            "alt_mode": "mode_b",
        },
    )
    monkeypatch.setattr(runtime.hybrid, "retrieve_hybrid", retrieve)
    runtime.functions = {"mode_a": retrieve}
    try:
        result = runtime(Question("q", "q", ("answer",)))
        assert result["blocks"][0]["block_id"] == "g"
        assert ("modes" in calls[0]) == (variant not in {"direct_only", "no_hybrid"})
        assert _reranker.mode() == (
            "first_stage_only" if variant == "first_stage_only" else previous_mode
        )
    finally:
        runtime.close()
    assert _reranker.mode() == previous_mode
    assert not reasoning._topic_keywords_disabled.get()
    _check_forbidden("entities")


def test_corpus_preflight(native):
    runtime = NativeRetriever()
    questions = [Question("q", "q", ("a",), ("g",), build_id="20260523")]
    try:
        assert runtime.validate_corpus(questions, "20260523")["gold_ids_checked"] == 1
        with pytest.raises(ValueError, match="does not match"):
            runtime.validate_corpus(questions, "wrong")
        runtime.neo4j.get_all_block_ids = lambda: set()
        with pytest.raises(ValueError, match="missing graph blocks"):
            runtime.validate_corpus(questions, "20260523")
    finally:
        runtime.close()


def test_hybrid_propagates_context_and_reports_submode_failure(monkeypatch):
    from agent_rag.retrieval import hybrid

    marker = ContextVar("test_marker", default=False)
    token = marker.set(True)

    def good(*args, **kwargs):
        assert marker.get()
        return {"blocks": []}

    def bad(*args, **kwargs):
        assert marker.get()
        raise RuntimeError("guard tripped")

    monkeypatch.setattr(hybrid, "_MODE_TO_FN", {"mode_a": good, "mode_b": bad})
    monkeypatch.setattr(hybrid, "collect_related_links", lambda *a: [])
    try:
        result = hybrid.retrieve_hybrid(
            "q", [1.0], None, None, None, modes=["mode_a", "mode_b"], skip_answer=True
        )
        assert "guard tripped" in result["evaluation_errors"]["mode_b"]
    finally:
        marker.reset(token)


def test_ablation_replay_runs_separate_conditions(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {"id": "q", "question": "q", "short_answers": ["answer"], "gold_block_ids": ["g"]},
            ]
        ),
        "utf-8",
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            [
                {"id": "q", "variant": "full", "blocks": [{"block_id": "g", "content": "answer"}]},
                {"id": "q", "variant": "direct_only", "blocks": []},
            ]
        ),
        "utf-8",
    )
    output = tmp_path / "reports"
    assert (
        main(
            [
                "--dataset",
                str(dataset),
                "--replay",
                str(predictions),
                "--ablate",
                "direct_only",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    comparison = json.loads((output / "comparison.json").read_text("utf-8"))
    assert comparison["metric_deltas_vs_full"]["direct_only"]["block_hit@5"] == -1
