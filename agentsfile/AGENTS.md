# AGENTS.md — Master Context File
## Production-Grade RAG Service

**Last Updated:** 2026-08-04  
**Current Phase:** Phase 2 — Retrieval Pipeline (in progress)  
**Overall Progress:** ~30% complete

> **ROLE OVERRIDE (active as of 2026-08-04):** This agent operates in **EXECUTION + EXPLANATION** mode, not mentor-only mode. It implements the requested module(s) and then explains in detail what it did, why, and the key decisions. This overrides any "mentor-guide / do-not-write" constraint elsewhere (e.g. `agent_mentor.md` or the "read fully before responding" note in earlier files) for the files the user asks it to build. User instruction wins.

---

## IMPORTANT: Read This File First

This file is the single source of truth for any agent (human or AI) picking up this project. It gives you a complete snapshot of where the project stands right now: what has been built, what the active implementation decisions are, and exactly what to do next. Do not start reading code until you have finished this file.

**After this file, read in order:**
1. [`architecture.md`](./architecture.md) — Full data-flow diagram, every component's role, all key architectural decisions made so far.
2. [`PRD.md`](./PRD.md) — Product requirements, functional specs, and definition of done for each phase.
3. [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) — Phase-by-phase task checklist with what is done and what is pending.

---

## Project Summary

A standalone, production-grade Retrieval-Augmented Generation (RAG) service. It ingests a corpus of documents, stores them as searchable vector embeddings in PostgreSQL, and answers natural-language queries by retrieving the most relevant chunks and handing them to an LLM.

The project is built **manually first** (no framework magic), then rebuilt with LangChain/LangGraph in Phase 5 to make the abstraction gap explicit. The key deliverable is a RAGAS evaluation suite with before/after benchmark numbers.

---

## Current Technology Stack

| Layer | Technology | Decision Rationale |
|-------|------------|--------------------|
| Web Framework | FastAPI (async) | Production-grade, forces proper async patterns |
| Language | Python 3.11 | Modern type hints, async support |
| Database | PostgreSQL + `pgvector` | Chosen over Qdrant to force manual RRF implementation (higher learning value) |
| Embeddings | Cohere API (`embed-english-v3.0`, 1024 dims) | Free Developer Tier; avoids needing local GPU |
| LLM Generation | Groq (provider TBD, e.g. Llama 3) | Free API, no cost for portfolio project |
| Dependency Mgmt | Poetry | Reproducible installs via `poetry.lock` |
| Containerization | Docker + docker-compose | Full local dev environment |
| ORM | SQLAlchemy 2.0 (async) with `asyncpg` | Modern async-native ORM |

---

## What Has Been Built (Completed Work)

### Phase 0: Scaffolding ✅
All boilerplate is in place:
- `Dockerfile` + `docker-compose.yml` (Postgres 16 with `pgvector/pgvector:pg16` image)
- `pyproject.toml` with all dependencies managed via Poetry
- `.env.example` / `.env.local` — all secrets via env vars, never hardcoded
- `src/main.py` — FastAPI app skeleton with `/health` route
- `src/config.py` — Pydantic `BaseSettings` loading from `.env.local` / `.env`

### Phase 1: Ingestion Pipeline ✅
The full ingestion pipeline is built and architecturally verified.

**Files completed:**
- [`src/ingestion/loaders.py`](../src/ingestion/loaders.py) — Reads files from `corpus/raw/`. Returns `list[{"text": str, "metadata": {"filename": str}}]`. Handles `.md`, `.txt`, and `.pdf` (text-native PDFs only — scanned image PDFs were rejected in favour of text corpus).
- [`src/ingestion/chunker.py`](../src/ingestion/chunker.py) — Fixed-size chunking with overlap. `chunk_text()` splits raw string. `chunk_documents()` wraps it, preserving metadata + adding `chunk_index`. Returns `list[{"text": str, "metadata": {..., "chunk_index": int}}]`. Chunk size: 500 chars, overlap: 50 chars.
- [`src/ingestion/embedder.py`](../src/ingestion/embedder.py) — Async embedder using Cohere `AsyncClientV2`. Batches texts into groups of 96, fans them out concurrently using `asyncio.gather` + `asyncio.Semaphore(10)`. Tenacity retries on `TooManyRequestsError`, `ServiceUnavailableError`, `InternalServerError` with exponential backoff + jitter (5 attempts max). Returns `list[list[float]]` in same order as input.
- [`src/ingestion/pipeline.py`](../src/ingestion/pipeline.py) — Orchestrator: loader → chunk → embed → save to DB. Uses `asyncio.to_thread` to offload synchronous file I/O. Deduplicates documents by `source_url`. Zips chunks + vectors together for DB insert. Per-document commit + rollback for fault tolerance (one failed document does not abort the whole corpus). Returns `IngestResult` dataclass.
- [`src/db/models.py`](../src/db/models.py) — Two SQLAlchemy ORM tables:
  - `Document` (id UUID, source_url, title, metadata JSONB, created_at)
  - `Chunk` (id UUID, document_id FK→Document CASCADE, text, chunk_index, `embedding Vector(1024)`, metadata JSONB, created_at)
- [`src/db/session.py`](../src/db/session.py) — Async SQLAlchemy session factory using `asyncpg`.

**Corpus (current):**
Three text-native `.md` files in `corpus/raw/`:
- `python-decorators-explained.md`
- `python-requests-handbook.md`
- `react-hooks-cheatsheet.md`

### Phase 2: Retrieval Pipeline 🔄 IN PROGRESS

**Dense retrieval is implemented and live-tested against the mock store** (results below). **BM25 sparse retrieval is also implemented and compared.** The remaining two retrieval files (fusion, reranker) are stubs.

**Storage backend decision (locked 2026-08-04):** No database is reachable on this machine right now (Docker not installed; native PG17 lacks pgvector). To keep building, retrieval runs against an **in-memory mock store** behind a swappable `VectorStore` interface:

- [`src/retrieval/stores.py`](../src/retrieval/stores.py) — The abstraction: `VectorStore` ABC + two backends:
  - `InMemoryVectorStore` (default, `VECTOR_STORE=inmemory`) — cosine similarity in pure Python over a snapshot (`data/mock_index.json`, built by `scripts/build_mock_index.py` from the same loader→chunker→embedder pipeline as real ingestion, or lazily from `corpus/raw/`). **No DB required.**
  - `PgVectorStore` (`VECTOR_STORE=pgvector`) — the real pgvector `<=>` cosine-distance SQL, ready for the Docker/Postgres path.
- [`src/retrieval/dense.py`](../src/retrieval/dense.py) — **DONE.** Embeds the query (`input_type="search_query"`) and delegates ranking to the configured store. No Postgres coupling.
- [`src/retrieval/sparse.py`](../src/retrieval/sparse.py) — **DONE.** BM25 (Okapi) over the same chunk records as dense (shared ids via `stores.load_chunk_records()`), so RRF can merge by chunk id. Tokenizer is a lowercase alphanumeric split.
- [`scripts/build_mock_index.py`](../scripts/build_mock_index.py) — Builds `data/mock_index.json` (29 chunks from the 3-file corpus).

**How to swap to pgvector later (intentional one-liner):**
```
docker-compose up -d            # once Docker is installed
poetry run python scripts/ingest_corpus.py
VECTOR_STORE=pgvector poetry run uvicorn src.main:app
```

**Docker is deferred, not dropped:** it remains an explicit Phase-6 learning goal (`docker-compose.yml`, Dockerfile, deployment already scaffolded). The mock layer exists precisely so the pipeline works and is testable *before* Docker is introduced, then swaps over cleanly.

**Smoke-test result (mock store, 2026-08-04):**
- `"how do I use a Python decorator?"` → top hit `python-decorators-explained.md chunk=0` (0.731)
- `"what is a session in the requests library?"` → `python-requests-handbook.md chunk=6` (0.521)
- `"how does useState work in React?"` → `react-hooks-cheatsheet.md chunk=2` (0.626)
- Whitespace-only query → `[]` (guard works)

Remaining Phase-2 stubs:
- [`src/retrieval/fusion.py`](../src/retrieval/fusion.py) — **STUB.** Must implement Reciprocal Rank Fusion (RRF) to merge dense + sparse results.
- [`src/retrieval/reranker.py`](../src/retrieval/reranker.py) — **STUB.** Must implement cross-encoder reranking.

---

## What Is NOT Built Yet

| Phase | Component | Status |
|-------|-----------|--------|
| 2 | Dense Retrieval (`dense.py`) | Done (mock store) |
| 2 | Vector Store Abstraction (`stores.py`) | Done (inmemory + pgvector backends) |
| 2 | Mock Index Snapshot (`data/mock_index.json`) | Done (29 chunks) |
| 2 | Sparse / BM25 Retrieval (`sparse.py`) | Done |
| 2 | Reciprocal Rank Fusion (`fusion.py`) | Stub only |
| 2 | Cross-Encoder Reranker (`reranker.py`) | Stub only |
| 2 | Prompt Builder (`generation/prompt_builder.py`) | Stub only |
| 2 | LLM Generator (`generation/generator.py`) | Stub only |
| 2 | Manual Pipeline Orchestrator (`pipeline/manual.py`) | Stub only |
| 2 | Query API route (`api/routes/query.py`) | Stub only |
| 3 | RAGAS Evaluation Harness (`evaluation/ragas_eval.py`) | Stub only |
| 3 | Test Set (`evaluation/test_set.json`) | Empty array |
| 4 | Observability / Langfuse tracing (`observability/tracing.py`) | Stub only |
| 5 | LangChain rebuild (`pipeline/langchain_pipeline.py`) | Stub only |
| 5 | LangGraph agentic pipeline (`pipeline/agentic.py`) | Stub only |
| 6 | Deployment | Not started |

---

## Active Architectural Decisions

These decisions are locked. Do not revisit them without a strong reason.

| Decision | Choice | Why |
|----------|--------|-----|
| Vector Store | PostgreSQL + pgvector | Forces manual RRF implementation; higher learning value than Qdrant |
| Embedding Model | Cohere `embed-english-v3.0` (1024 dims) | Free tier; no local GPU needed |
| Chunking Strategy | Fixed-size, 500 chars, 50 char overlap | Baseline; prevents context being severed at boundary |
| Corpus Format | Text-native Markdown files | Scanned PDFs were attempted and rejected (OCR path rejected to keep focus on RAG) |
| Async Strategy | `asyncio.gather` + `Semaphore` for embedding | Concurrency for network I/O; avoids sequential API calls |
| LLM Provider | Groq (free API) | No cost; Llama 3 models available |

---

## Decisions Still Open (Must Resolve Before Each Phase)

1. **Docker / Postgres rollout:** Deferred by decision on 2026-08-04 (Docker not installed; mock store keeps the pipeline moving). Must be revisited before Phase 4/6 — the `PgVectorStore` backend is ready and tested at SQL-compile level, and `docker-compose.yml` + `scripts/ingest_corpus.py` are already in place.
2. **Observability tool:** Langfuse (self-hosted, open source) vs. LangSmith (managed). Decide before Phase 4.
3. **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` is the default. Confirm before Phase 2 reranker step.
4. **Deployment target:** Railway, Render, Fly.io, or minimal AWS. Decide before Phase 6.

---

## How to Run (Current State)

**With the mock store (no database needed — the working path right now):**
```bash
# 1. Install dependencies
poetry install

# 2. Copy env and fill in COHERE_KEY
cp .env.example .env.local

# 3. (Optional) pre-build the mock index snapshot — embeds the corpus once
poetry run python scripts/build_mock_index.py

# 4. Run the API (VECTOR_STORE defaults to inmemory)
poetry run uvicorn src.main:app --reload
# GET http://localhost:8000/health → {"status": "ok"}
```

**With Postgres + pgvector (the Docker path, deferred until Docker is installed):**
```bash
# 1. Start Postgres with pgvector
docker-compose up -d

# 2. Run ingestion to fill the chunk table with embeddings
poetry run python scripts/ingest_corpus.py

# 3. Switch the retrieval backend and run
VECTOR_STORE=pgvector poetry run uvicorn src.main:app --reload
```

---

## Immediate Next Action

Dense and sparse retrieval are complete and verified. The next task is **Reciprocal Rank Fusion (RRF)** in [`src/retrieval/fusion.py`](../src/retrieval/fusion.py):

1. Implement `async fuse(dense_results, sparse_results, k=60) -> list[tuple[Chunk, float]]`.
2. Formula: `score(d) = sum over each list of 1 / (k + rank(d))`, where rank is 1-indexed position; merge by chunk id.
3. Merge same-ids across both lists; documents present in only one list still score via that list alone.
4. After fusion, implement `reranker.py` (cross-encoder) to complete the manual retrieval pipeline.

---

## Update Log

| Date | Change |
|------|--------|
| 2026-07-31 | Project initialized. PRD, IMPLEMENTATION_PLAN, README, folder structure, boilerplate all created. |
| 2026-08-03 | Phase 1 complete. Loader, chunker, embedder, DB models, ingestion pipeline all built and verified. Corpus switched from scanned PDFs to text-native Markdown files. Embedding provider changed from OpenAI to Cohere (free tier). |
| 2026-08-04 | AGENTS.md and ARCHITECTURE.md created. Retrieval stubs confirmed as not yet implemented. |
| 2026-08-04 | Dense retrieval implemented and live-tested via new `VectorStore` abstraction (mock in-memory backend). Docker deferred (not installed), pgvector backend kept ready behind `VECTOR_STORE` config. Role set to EXECUTION + EXPLANATION. |
| 2026-08-04 | BM25 sparse retrieval implemented and compared against dense. Dense won on all 4 natural-language test queries; BM25 noise at 29-chunk corpus demonstrates why RRF hybrid retrieval matters. `rank-bm25` installed. |
