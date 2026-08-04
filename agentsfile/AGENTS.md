# AGENTS.md — Master Context File
## Production-Grade RAG Service

**Last Updated:** 2026-08-04  
**Current Phase:** Phase 2 — Retrieval Pipeline (in progress)  
**Overall Progress:** ~30% complete

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

**The Role of the Mentor Agent (Antigravity):** This project is a teaching exercise. The agent acts as a mentor/architect — it explains concepts, reviews code, and asks questions. It does NOT write the core retrieval logic, evaluation harness, or pipeline logic unprompted. See [`../agent_mentor.md`](../agent_mentor.md) for the full mentor brief.

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
All four retrieval files exist as stubs with clear implementation comments. **None of the retrieval logic has been implemented yet.**

- [`src/retrieval/dense.py`](../src/retrieval/dense.py) — **STUB.** Must implement async vector search via `pgvector` `<=>` cosine distance operator.
- [`src/retrieval/sparse.py`](../src/retrieval/sparse.py) — **STUB.** Must implement BM25 keyword search using `rank_bm25` in-memory.
- [`src/retrieval/fusion.py`](../src/retrieval/fusion.py) — **STUB.** Must implement Reciprocal Rank Fusion (RRF) to merge dense + sparse results.
- [`src/retrieval/reranker.py`](../src/retrieval/reranker.py) — **STUB.** Must implement cross-encoder reranking.

---

## What Is NOT Built Yet

| Phase | Component | Status |
|-------|-----------|--------|
| 2 | Dense Retrieval (`dense.py`) | Stub only |
| 2 | Sparse / BM25 Retrieval (`sparse.py`) | Stub only |
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

1. **Observability tool:** Langfuse (self-hosted, open source) vs. LangSmith (managed). Decide before Phase 4.
2. **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` is the default. Confirm before Phase 2 reranker step.
3. **Deployment target:** Railway, Render, Fly.io, or minimal AWS. Decide before Phase 6.

---

## How to Run (Current State)

```bash
# 1. Start Postgres with pgvector
docker-compose up -d

# 2. Install dependencies
poetry install

# 3. Copy env and fill in secrets
cp .env.example .env.local
# Set DATABASE_URL and COHERE_KEY in .env.local

# 4. Run ingestion (requires DB to be up and corpus in corpus/raw/)
poetry run python scripts/ingest_corpus.py

# 5. Start the API
poetry run uvicorn src.main:app --reload
# GET http://localhost:8000/health → {"status": "ok"}
```

---

## Immediate Next Action

The next task is implementing **Dense Retrieval** in [`src/retrieval/dense.py`](../src/retrieval/dense.py).

This function must:
1. Accept a query string and a `top_k` integer.
2. Use the embedder to convert the query string into a vector.
3. Query Postgres using the `<=>` cosine distance operator via pgvector to find the `top_k` most similar `Chunk` rows.
4. Return the results as a list of `Chunk` objects with their scores.

**Do NOT implement this file without the student/developer writing the core SQL/ORM query themselves.** Present the concept, explain cosine distance, and ask them to attempt the query first.

---

## Update Log

| Date | Change |
|------|--------|
| 2026-07-31 | Project initialized. PRD, IMPLEMENTATION_PLAN, README, folder structure, boilerplate all created. |
| 2026-08-03 | Phase 1 complete. Loader, chunker, embedder, DB models, ingestion pipeline all built and verified. Corpus switched from scanned PDFs to text-native Markdown files. Embedding provider changed from OpenAI to Cohere (free tier). |
| 2026-08-04 | AGENTS.md and ARCHITECTURE.md created. Retrieval stubs confirmed as not yet implemented. |
