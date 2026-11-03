"""Run with python -m agent_rag.evaluation; no model calls in validation/replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from agent_rag.evaluation.core import evaluate, load_questions
from agent_rag.evaluation.runtime import VARIANTS


def load_replay(path: str, variant: str) -> dict:
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    records = {}
    for row in rows:
        # Missing variant denotes the full-system condition only.
        if row.get("variant", "full") != variant:
            continue
        qid = row["id"]
        if qid in records:
            raise ValueError(f"Duplicate replay id for {variant}: {qid}")
        if not isinstance(row.get("blocks"), list):
            raise ValueError(f"Replay record {qid} is missing blocks")
        records[qid] = row
    if not records:
        raise ValueError(f"No replay records for variant {variant}")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PolyUQuest retrieval evaluation / ablation")
    parser.add_argument("--dataset", default="data/dataset_polyu_300.json")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument(
        "--ablate",
        nargs="+",
        choices=VARIANTS,
        help="Run full plus these variants in separate processes",
    )
    parser.add_argument("--replay", help="JSON/JSONL predictions; no services or model calls")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--corpus-build-id", help="Required for live evaluation")
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args(argv)
    if not args.ks or min(args.ks) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Cutoffs and limit must be positive")
    questions = load_questions(args.dataset)
    if args.tags:
        questions = [q for q in questions if set(q.tags) & set(args.tags)]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        parser.error("No questions selected")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "n_queries": len(questions),
                    "tags": sorted({t for q in questions for t in q.tags}),
                    "build_ids": sorted({q.build_id for q in questions if q.build_id}),
                    "n_with_gold": sum(bool(q.gold_block_ids) for q in questions),
                },
                indent=2,
            )
        )
        return 0
    if not args.replay and not args.corpus_build_id:
        parser.error("Live evaluation requires --corpus-build-id for the indexed snapshot")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.ablate:
        (output / "comparison.json").unlink(missing_ok=True)
        variants = list(dict.fromkeys(["full", *args.ablate]))
        reports = {}
        for variant in variants:
            command = [
                sys.executable,
                "-m",
                "agent_rag.evaluation",
                "--dataset",
                args.dataset,
                "--variant",
                variant,
                "--output-dir",
                str(output),
                "--ks",
                *map(str, args.ks),
            ]
            for flag, value in (
                ("--replay", args.replay),
                ("--corpus-build-id", args.corpus_build_id),
                ("--limit", args.limit),
            ):
                if value is not None:
                    command += [flag, str(value)]
            if args.tags:
                command += ["--tags", *args.tags]
            # A failed condition aborts comparison; never reuse a stale report.
            subprocess.run(command, check=True)
            reports[variant] = json.loads((output / f"{variant}.json").read_text("utf-8"))
        base = reports["full"]["summary"]["metrics"]
        comparisons = {
            v: {key: value - base[key] for key, value in report["summary"]["metrics"].items()}
            for v, report in reports.items()
            if v != "full"
        }
        comparison = {
            "metric_deltas_vs_full": comparisons,
            "note": "Paired dataset; latency depends on cache/order, not a cold benchmark",
        }
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2), "utf-8")
        print(json.dumps(comparison, indent=2))
        return 0

    metadata = {
        "variant": args.variant,
        "backend": "replay" if args.replay else "native",
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "question_ids": [q.id for q in questions],
        "ks": sorted(set(args.ks)),
    }
    if args.replay:
        records = load_replay(args.replay, args.variant)
        missing = {q.id for q in questions} - records.keys()
        if missing:
            raise ValueError(f"Replay is incomplete: {len(missing)} missing question ids")
        report = evaluate(questions, lambda q: records[q.id], ks=tuple(sorted(set(args.ks))))
        metadata["predictions_sha256"] = hashlib.sha256(Path(args.replay).read_bytes()).hexdigest()
        metadata["latency_scope"] = "local scoring/replay overhead only"
    else:
        from agent_rag.evaluation.runtime import NativeRetriever

        runtime = NativeRetriever(args.variant)
        try:
            metadata.update(runtime.metadata)
            metadata["corpus"] = runtime.validate_corpus(questions, args.corpus_build_id)
            report = evaluate(questions, runtime, ks=tuple(sorted(set(args.ks))))
            metadata["usage"] = runtime.usage()
        finally:
            runtime.close()
        metadata["latency_scope"] = "routing + embedding + retrieval, no answer generation"
    report["metadata"] = metadata
    path = output / f"{args.variant}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {path}")
    return int(report["summary"]["n_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
