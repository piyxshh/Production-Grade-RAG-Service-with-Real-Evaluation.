# AGENTS.md — Master Context File
## Production-Grade RAG Service

**Last Updated:** 2026-08-14  
**Current Phase:** Phase 3 — Evaluation & Benchmarking Suite (✅ COMPLETED)  
**Overall Progress:** ~75% complete

> **ROLE OVERRIDE (active as of 2026-08-04):** This agent operates in **EXECUTION + EXPLANATION** mode, not mentor-only mode. It implements the requested module(s) and then explains in detail what it did, why, and the key decisions. This overrides any "mentor-guide / do-not-write" constraint elsewhere (e.g. `agent_mentor.md` or the "read fully before responding" note in earlier files) for the files the user asks it to build. User instruction wins.

---

## IMPORTANT: Read This File First

This file is the single source of truth for any agent (human or AI) picking up this project. It gives you a complete snapshot of where the project stands right now: what has been built, what the active implementation decisions are, and exactly what to do next. Do not start reading code until you have finished this file.

---

## What Has Been Built (Completed Work)

### Phase 0: Scaffolding ✅
- `Dockerfile` + `docker-compose.yml` (Postgres 16 with `pgvector/pgvector:pg16` image)
- `pyproject.toml` with all dependencies managed via Poetry
- `.env.example` / `.env.local` — all secrets via env vars, never hardcoded
- `src/main.py` — FastAPI app skeleton with `/health` route
- `src/config.py` — Pydantic `BaseSettings` loading from `.env.local` / `.env`

### Phase 1: Ingestion Pipeline ✅
- [`src/ingestion/loaders.py`](../src/ingestion/loaders.py) — Ingests `.md`, `.txt`, and `.pdf`.
- [`src/ingestion/chunker.py`](../src/ingestion/chunker.py) — Fixed-size chunking (500 chars) with 50-char overlap.
- [`src/ingestion/embedder.py`](../src/ingestion/embedder.py) — Async embedder using Cohere `embed-english-v3.0` (1024 dims), batching (96), semaphore (10), tenacity retries.
- [`src/ingestion/pipeline.py`](../src/ingestion/pipeline.py) — Ingestion orchestrator with per-document fault isolation and deduplication.
- [`src/db/models.py`](../src/db/models.py) — `Document` and `Chunk` SQLAlchemy ORM models with `Vector(1024)` support.
- [`src/db/session.py`](../src/db/session.py) — Async session factory.

### Phase 2: Manual Retrieval & Generation Pipeline ✅
- [`src/retrieval/stores.py`](../src/retrieval/stores.py) — `VectorStore` abstraction with `InMemoryVectorStore` (`data/mock_index.json`) and `PgVectorStore`.
- [`src/retrieval/dense.py`](../src/retrieval/dense.py) — Async semantic vector search.
- [`src/retrieval/sparse.py`](../src/retrieval/sparse.py) — BM25 Okapi lexical search using `rank_bm25`.
- [`src/retrieval/fusion.py`](../src/retrieval/fusion.py) — Reciprocal Rank Fusion (RRF, $k=60$) merging dense and sparse candidates.
- [`src/retrieval/reranker.py`](../src/retrieval/reranker.py) — Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with fallback.
- [`src/generation/prompt_builder.py`](../src/generation/prompt_builder.py) — Strictly grounded prompt builder with refusal string and citation rules.
- [`src/generation/generator.py`](../src/generation/generator.py) — Async LLM caller targeting Groq (`llama-3.1-8b-instant`) and OpenAI.
- [`src/pipeline/manual.py`](../src/pipeline/manual.py) — End-to-end pipeline orchestrator.
- [`src/api/routes/query.py`](../src/api/routes/query.py) — `POST /query` endpoint with Pydantic schemas.
- **Tests:** 16 unit and integration tests passing (`pytest tests/ -v`).

### Phase 3: Evaluation & Benchmarking Suite ✅
- [`evaluation/test_set.json`](../evaluation/test_set.json) — 50 hand-crafted test questions across 5 categories with exact ground truth chunk IDs.
- [`src/evaluation/retrieval_eval.py`](../src/evaluation/retrieval_eval.py) — Retrieval ablation measuring Hit@K, Recall@K, MRR, and NDCG@K across Dense, BM25, Hybrid RRF, and Hybrid + Reranker.
- [`src/evaluation/generation_eval.py`](../src/evaluation/generation_eval.py) — Evaluates refusal accuracy, citation precision, token F1, and factual grounding.
- [`src/evaluation/latency_eval.py`](../src/evaluation/latency_eval.py) — Measures per-stage and end-to-end latency percentiles (Mean, P50, P90, P95, Max).
- [`scripts/evaluate.py`](../scripts/evaluate.py) & [`scripts/run_eval.py`](../scripts/run_eval.py) — CLI evaluation runners producing ASCII tables and exporting to `evaluation/results/evaluation_report.json` and `evaluation/results/retrieval_ablation.csv`.

---

## What Is NOT Built Yet

| Phase | Component | Status |
|-------|-----------|--------|
| 4 | Observability / Langfuse tracing (`src/observability/tracing.py`) | Stub only |
| 5 | LangChain rebuild (`src/pipeline/langchain_pipeline.py`) | Stub only |
| 5 | LangGraph agentic pipeline (`src/pipeline/agentic.py`) | Stub only |
| 6 | Deployment | Not started |

---

## How to Run (Current State)

**1. Run test suite (16 tests):**
```bash
python -m pytest tests/ -v
```

**2. Run full 50-query evaluation suite:**
```bash
python scripts/evaluate.py
```

**3. Run the FastAPI service:**
```bash
python -m uvicorn src.main:app --reload
# GET  http://localhost:8000/health
# POST http://localhost:8000/query
```

---

## Update Log

| Date | Change |
|------|--------|
| 2026-07-31 | Project initialized. PRD, IMPLEMENTATION_PLAN, README, folder structure, boilerplate created. |
| 2026-08-03 | Phase 1 complete. Loader, chunker, embedder, DB models, ingestion pipeline built. |
| 2026-08-04 | Dense and sparse retrieval implemented with VectorStore abstraction (mock in-memory backend). |
| 2026-08-14 | **Phase 2 completed.** Implemented RRF (`fusion.py`), Cross-Encoder (`reranker.py`), Prompt Builder (`prompt_builder.py`), Groq generator (`generator.py`), manual pipeline (`manual.py`), and `POST /query`. 16 unit/integration tests passing. |
| 2026-08-14 | **Phase 3 completed.** Created 50-question evaluation dataset (`test_set.json`), retrieval ablation module (`retrieval_eval.py`), generation quality evaluator (`generation_eval.py`), latency profiler (`latency_eval.py`), and automated CLI runner (`scripts/evaluate.py`). Real empirical metrics exported to CSV/JSON and documented. |


