# PolyUQuest evaluation and ablation

The `agent_rag.evaluation` module evaluates retrieval over the PolyUQuest benchmark
and measures the contribution of the repository's retrieval components. It uses
the configured query router and native mode A/B/C and hybrid retrieval functions.
The final answer-generation step is disabled for every experimental condition.

## Module structure

| File | Responsibility |
|---|---|
| `evaluation/core.py` | Dataset validation, block and answer-span metrics, per-tag aggregation |
| `evaluation/runtime.py` | Native retrieval dispatch, component switches, corpus preflight |
| `evaluation/__main__.py` | CLI, independent ablation processes, prediction replay and local reports |
| `tests/test_evaluation.py` | Service-free verification of scoring and experiment execution |

The default dataset is `data/dataset_polyu_300.json`, a JSON array of 300 annotated
questions across `single_hop`, `multi_page`, `entity_list`, `mixed_ab` and `mixed_ac`.
Each question must provide `id`, `question`, `short_answers` and `gold_block_ids`.
`short_answers` and `gold_block_ids` are arrays of nonempty strings; at least one
must contain annotations. `tags` supplies analysis groups and `_build_id` identifies
the annotated corpus build. Additional dataset provenance fields are not used to
retrieve evidence. IDs must be unique; invalid records fail validation.

Retrieval returns an ordered `blocks` list with `block_id`, `content` and source
metadata. Gold block annotations determine retrieval relevance; `short_answers`
provide supplementary answer-span coverage measurements.

## Run locally

Run from the repository root using the installed project Python environment.
For a fresh environment, install with `python -m pip install -e ".[dev]"`.
The examples below use `python`; on Windows the existing environment can be used
as `.venv\Scripts\python.exe`.

Validate the public dataset without databases, API credentials or model calls:

```console
python -m agent_rag.evaluation --validate-only
```

The released 300 questions reference build `20260523`. Live runs require the
matching indexed Neo4j and Qdrant stores, configured `.env` credentials and model
services. The release does not contain that corpus or its offline build pipeline.
Before retrieval, the harness checks every selected gold block ID is present in
both stores. A mismatched declared build ID or missing gold blocks aborts the run.
The build ID is declared by the operator: ID resolution alone does not verify
the identity, content or completeness of the entire corpus snapshot.

```console
python -m agent_rag.evaluation --corpus-build-id 20260523 --limit 5
python -m agent_rag.evaluation --corpus-build-id 20260523 --tags multi_page mixed_ab
python -m agent_rag.evaluation --corpus-build-id 20260523 --ablate direct_only no_entity_layer no_topic_keywords no_hybrid first_stage_only
```

These are retrieval evaluations (`skip_answer=True`), not generated-answer
quality tests. Live retrieval still uses paid models for routing, embeddings,
query rewriting, decomposition, keyword extraction and reranking as configured.
The harness does not launch live runs as part of its tests.

## Ablation definitions

| Variant | Exact intervention |
|---|---|
| `full` | Native auto router and configured retrieval behavior, without final answer generation |
| `direct_only` | Bypass router and force mode A, setting router confidence to 1.0; this also affects confidence-gated rewrites, so it is a forced-A baseline, not a pure graph ablation |
| `no_hybrid` | Keep the router decision and confidence but dispatch only its primary mode |
| `no_entity_layer` | Route with allowed modes A/B only, forbid entity/relation/topic-keyword Qdrant collection access; retain page/block layers, including DOM hierarchy and page links |
| `no_topic_keywords` | Disable mode C's high-level keyword extraction and relation-ANN branch, retaining entity ANN and graph expansion |
| `first_stage_only` | Use the existing reranker `first_stage_only` path, retaining candidate generation and final cap while skipping the cross-encoder |

`--ablate` runs `full` plus the requested conditions in separate Python processes
over the same questions and cutoffs. No YAML configuration is edited. Variant
controls are evaluation-only; no gold answers, gold blocks or `gold_mode` are
passed into retrieval. Mixed conditions inherit the configured auto-hybrid gate;
they do not force a hybrid route just because the question has a mixed tag.

The current YAML has `retrieval.router.disable_hybrid: true`: `no_hybrid` can
therefore be identical to `full`. `no_topic_keywords` has no intervention on
questions not using mode C. An inactive component need not show a metric delta.
`first_stage_only` requires the reranker path to be enabled to have an effect.

Hybrid workers inherit context variables, preserving TopicKeyword switches,
collection guards, query-rewrite gating and cost stages. Sub-mode failures are
attached to native hybrid results and treated as evaluation errors, so a guard
violation or partial service failure cannot masquerade as a successful run.

## Metrics and outputs

- `block_hit@k`: any gold block retrieved in the first k unique returned blocks.
- `block_mrr@k`: reciprocal rank of the first gold block within k (0 on a miss).
- `block_recall@k`: fraction of annotated gold block IDs retrieved within k.
- `answer_hit@k`: at least one normalized answer phrase found within an individual
  block's content. A text match is a heuristic diagnostic, not proof of correctness.
- `answer_mrr@k`: reciprocal rank of the first phrase-matching block within k.
- `answer_coverage@k`: fraction of the answer spans found across the first k
  blocks, useful for questions requiring multiple facts or list items.

Answer matching normalizes Unicode, case and whitespace. Phrase boundaries avoid
matching e.g. `2026` inside `12026`. Answer phrases must occur in order.
There is no generated-answer EM/F1 or faithfulness claim. Metrics use returned
order without re-sorting. Duplicate block IDs are deduplicated before scoring.
Queries with no gold IDs are excluded from block-metric denominators rather than
counted as misses. Every metric includes its denominator; breakdowns are by tag.
Failures remain in applicable denominators with zero scores and an explicit error.

Outputs are local under `results/evaluation/` (ignored by Git), containing selected
question IDs, dataset hash, effective retrieval settings, per-query results and
errors, per-tag summaries and paired metric deltas. Nonzero error counts produce
a nonzero exit code; ablation comparison stops on a failed condition. Existing
disk caches are retained and router cache is cleared per run. Latency is warm/
mixed-cache dependent, includes routing/embedding/retrieval, and excludes preflight
and answer generation. Do not use variant ordering to claim cold latency gains.
Reported LLM/reranker usage counters exclude embedding usage and monetary cost.

## Offline replay and tests

Replay consumes a JSON array or JSONL of native-shaped records:

```json
{"id":"q_single_hop_001","variant":"full","blocks":[{"block_id":"actual-block-id","content":"retrieved passage","source_url":"https://www.polyu.edu.hk/"}]}
```

Include every selected question ID. `variant` defaults to `full`; supply a
distinct set of records per condition for ablation replay. Replay scores supplied
outputs; it does not execute or verify the claimed retrieval intervention, corpus
snapshot or provenance. Its latency is local replay/scoring overhead only.

```console
python -m agent_rag.evaluation --replay predictions.jsonl --output-dir results/evaluation/replay
python -m pytest tests/test_evaluation.py -q
```

The service-free tests cover the public schema, scoring, denominator handling, errors,
replay, native dispatch, component switches, corpus preflight and hybrid context
propagation. Passing them establishes implementation checks, not a live 300-query
benchmark result.
