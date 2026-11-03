# PolyUQuest

**PolyUQuest** is a structure-aware, graph-enhanced Retrieval-Augmented Generation (RAG) system for university websites. It combines a three-layer knowledge graph (WebPage → DOM Block → Entity) with hybrid dense/sparse retrieval and a confidence-aware query router to answer natural-language questions over a website's content.

This repository contains the **online serving stack**: a FastAPI backend and a Next.js frontend. The two share a Neo4j (graph) + Qdrant (vector) backing store.

---

## Architecture at a glance

```
┌────────────────┐     HTTP /api      ┌──────────────────────┐
│  Next.js UI    │ ─────────────────► │  FastAPI backend     │
│  (frontend/)   │ ◄───────────────── │  (src/agent_rag/)    │
└────────────────┘                    └──────────┬───────────┘
                                                  │
                                   ┌──────────────┴──────────────┐
                                   ▼                             ▼
                            ┌────────────┐               ┌──────────────┐
                            │   Neo4j    │               │    Qdrant    │
                            │  (graph)   │               │  (vectors)   │
                            └────────────┘               └──────────────┘
```

**Query path** (`src/agent_rag/retrieval/`):
- `router.py` — confidence-aware router picks a retrieval mode.
- `direct.py` — block-level ANN search (mode A).
- `navigation.py` — page-anchored search with link expansion (mode B).
- `reasoning.py` — entity + topic-keyword graph traversal (mode C).
- `hybrid.py` — combines modes when router confidence is low.

---

## Prerequisites

- **Python ≥ 3.11** — managed via [uv](https://docs.astral.sh/uv/) (recommended).
- **Node.js ≥ 18** + npm — for the frontend.
- **Docker** + Docker Compose — for Neo4j and Qdrant.
- An LLM API key and an embedding API key (see `.env.example`). The defaults use [SiliconFlow](https://siliconflow.cn/) for both the LLM and BGE-M3 embeddings, so no local GPU is required.

---

## Quick start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in at least:
#   SILICONFLOW_API_KEY   (LLM + embeddings)
#   NEO4J_PASSWORD        (must match docker-compose.yml)
```

### 2. Start the backing stores

```bash
docker-compose up -d
# Neo4j browser → http://localhost:7474  (bolt on 7687)
# Qdrant REST   → http://localhost:6333
```

### 3. Run the backend (FastAPI)

```bash
# With uv (auto-manages Python 3.11 + .venv):
uv sync
uv run agent-rag-serve          # → http://localhost:8000

# Swagger docs at http://localhost:8000/docs
```

<details>
<summary>Without uv (plain pip)</summary>

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
agent-rag-serve
```
</details>

### 4. Run the frontend (Next.js)

```bash
cd frontend
npm install
npm run dev                     # → http://localhost:3000
```

The frontend talks to the backend at `http://localhost:8000/api` by default.
To point it elsewhere, set `NEXT_PUBLIC_API_URL` before `npm run dev`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api npm run dev
```

---

## Configuration

| Concern | File |
|---|---|
| API keys / DB password / CORS | `.env` (template in `.env.example`) |
| Per-stage LLM model + temperature | `configs/llm.yaml` |
| Router confidence cutoffs, retrieval thresholds | `configs/thresholds.yaml` |
| Entity alias dictionary | `configs/aliases.yaml` |
| Crawler seeds / URL filter | `configs/crawl.yaml` |

**LLM provider** defaults to `siliconflow`; `deepseek` and `qwen` also work by
setting `LLM_PROVIDER` in `.env`. **Embeddings** default to BGE-M3 (1024-d) via
the SiliconFlow API.

---

## API endpoints

Once the backend is running, the main routes (all under `/api`) are:

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/query` | Ask a question; returns answer + retrieval trace |
| `GET`  | `/api/graph/stats` | Knowledge-graph node counts |
| `GET`  | `/api/health` | Liveness check |

Full interactive documentation is at `http://localhost:8000/docs`.

---

## Note on data

Retrieval evaluation and component ablations are available through
`python -m agent_rag.evaluation`. See [the evaluation guide](docs/EVALUATION.md)
for dataset validation, corpus requirements, metrics, variants and offline replay.

This repository ships the **serving code** and the **PolyU benchmark dataset**.
The Neo4j + Qdrant stores start empty — you need to populate them with an indexed
corpus before live queries return results. The offline indexing pipeline is not
included here; the released dataset lives under `data/` —
see [`data/README.md`](data/README.md).

---

## License

This project is **dual-licensed**:

- **Code** (everything except `data/`) — [Apache License 2.0](LICENSE).
- **Dataset** (`data/`) — **research use only**; derived from public PolyU web
  pages with required `_source_urls` attribution. See
  [`data/DATA_LICENSE.md`](data/DATA_LICENSE.md).

## Project layout

```
PolyUQuest/
├── src/agent_rag/        # FastAPI backend + retrieval engine
│   ├── api/              # routes, schemas, app entry point
│   ├── retrieval/        # router + retrieval modes (direct/nav/reasoning/hybrid)
│   ├── storage/          # Neo4j + Qdrant adapters
│   ├── kg/               # knowledge-graph extraction & resolution
│   ├── crawler/          # web crawler + URL filter
│   ├── html_processing/  # DOM → block-tree cleaner
│   └── llm/              # LLM client + Jinja prompt templates
├── frontend/             # Next.js 14 UI
│   ├── app/              # routes (chat, graph, compare)
│   ├── components/       # React components
│   └── lib/              # API client + state store
├── configs/              # YAML config (LLM, thresholds, aliases, crawl)
├── docker-compose.yml    # Neo4j + Qdrant
├── pyproject.toml        # Python deps (managed by uv)
└── .env.example          # environment template
```
