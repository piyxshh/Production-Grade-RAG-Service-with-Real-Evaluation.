# ARCHITECTURE.md — System Architecture
## Production-Grade RAG Service

**Last Updated:** 2026-08-14  
**Current Phase:** Phase 2 — Manual Retrieval & Generation Pipeline (✅ COMPLETED)

---

## Overview

This service answers natural-language questions by:
1. **Ingesting** a corpus of documents — loading, chunking, embedding, and storing them in Postgres (or in-memory mock store for zero-infra local dev).
2. **Retrieving** the most relevant chunks using parallel hybrid search (dense vector semantic search via Cohere + sparse keyword search via BM25 Okapi).
3. **Fusing** dense and sparse ranked candidates using Reciprocal Rank Fusion (RRF with $k=60$).
4. **Reranking** fused candidates with a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for fine-grained token-level cross-attention.
5. **Prompt Building** assembling strictly grounded context blocks with document titles and chunk index citations.
6. **Generating** grounded answers with inline citations using Groq (`llama-3.1-8b-instant`) or OpenAI.
7. **Serving** the pipeline via async FastAPI (`POST /query`).

---

## Full Data Flow Diagram

```
════════════════════════════════════════════════════════════════
 PHASE 1: INGESTION (runs once, offline)
════════════════════════════════════════════════════════════════

  corpus/raw/*.md
        │
        ▼
┌──────────────┐
│   Loader     │  loaders.py
│              │  Reads files → list[{text, metadata}]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Chunker    │  chunker.py
│              │  Fixed-size (500 chars) + overlap (50 chars)
│              │  → list[{text, metadata:{filename, chunk_index}}]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Embedder   │  embedder.py
│              │  Cohere embed-english-v3.0 (1024 dims)
│              │  Async batching (96/batch) + Semaphore(10)
│              │  → list[list[float]]  ← one vector per chunk
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│   PostgreSQL + pgvector      │  db/models.py + db/session.py
│                              │  (or data/mock_index.json for in-memory store)
│  ┌─────────────────────┐    │
│  │ Document Table       │    │
│  │ id, source_url,      │    │
│  │ title, metadata,     │    │
│  │ created_at           │    │
│  └──────────┬──────────┘    │
│             │ 1:Many FK      │
│  ┌──────────▼──────────┐    │
│  │ Chunk Table          │    │
│  │ id, document_id,     │    │
│  │ text, chunk_index,   │    │
│  │ embedding Vector(1024)│   │
│  │ metadata, created_at │    │
│  └─────────────────────┘    │
└──────────────────────────────┘


════════════════════════════════════════════════════════════════
 PHASE 2: RETRIEVAL & GENERATION PIPELINE (runs per query, online)
════════════════════════════════════════════════════════════════

  User Question (POST /query)
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ Dense Search │          │ Sparse Search│
│  dense.py    │          │  sparse.py   │
│              │          │              │
│ 1. Embed the │          │ 1. Tokenize  │
│    query via │          │    query     │
│    Cohere    │          │ 2. BM25      │
│ 2. Cosine    │          │    keyword   │
│    distance  │          │    search    │
│    in store  │          │    (rank_    │
│              │          │    bm25)     │
│ Returns top-K│          │ Returns top-K│
│ Chunk rows   │          │ Chunk rows   │
└──────┬───────┘          └──────┬───────┘
       │                          │
       └────────────┬─────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Fusion (RRF)    │  fusion.py
         │                  │
         │  Reciprocal Rank │
         │  Fusion formula: │
         │  score(d) =      │
         │  Σ 1/(k + rank)  │
         │  Merges two lists│
         │  into one ranked │
         │  list (k=60)     │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Reranker        │  reranker.py
         │                  │
         │  Cross-encoder   │
         │  ms-marco-       │
         │  MiniLM-L-6-v2   │
         │  Re-scores top-N │
         │  candidates for  │
         │  token precision │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Prompt Builder  │  generation/prompt_builder.py
         │                  │
         │  Assembles:      │
         │  - Strict system │
         │    grounding rules
         │  - Context chunks│
         │    with sources  │
         │  - User question │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  LLM Generator   │  generation/generator.py
         │                  │
         │  Groq API        │
         │  (Llama 3.1) /   │
         │  OpenAI Async    │
         └────────┬─────────┘
                  │
                  ▼
       Grounded Answer + Citations
       (QueryResponse schema)
```

---

## Component Descriptions

### `src/ingestion/loaders.py`
**Promise:** Uniformly ingests documents from `corpus/raw/` into `list[dict]` structure.  
**Handles:** `.md`, `.txt`, `.pdf`.

---

### `src/ingestion/chunker.py`
**Promise:** Splits documents into 500-character chunks with 50-character overlap to preserve semantic continuity across chunk boundaries.

---

### `src/ingestion/embedder.py`
**Promise:** Async embedder utilizing Cohere `embed-english-v3.0` (1024 dims). Batches into 96 chunks, bounded by `asyncio.Semaphore(10)` concurrency, with `tenacity` exponential retry backoff.

---

### `src/db/models.py` & `src/retrieval/stores.py`
**Promise:** Database models for `Document` and `Chunk` backed by pgvector, abstracted behind a `VectorStore` interface supporting `InMemoryVectorStore` (`data/mock_index.json`) and `PgVectorStore`.

---

### `src/retrieval/dense.py` ✅
**Promise:** Computes query embedding using Cohere (`input_type="search_query"`) and performs cosine similarity search against the active vector store backend.

---

### `src/retrieval/sparse.py` ✅
**Promise:** In-memory lexical search using `rank_bm25` (BM25Okapi). Operates over the same chunk corpus IDs to ensure lossless key matching during fusion.

---

### `src/retrieval/fusion.py` ✅
**Promise:** Merges dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).  
**Formula:** $\text{score}(d) = \sum \frac{1}{k + \text{rank}(d)}$ with $k=60$.  
**Why RRF:** Solves the problem of merging incomparable score distributions (cosine similarity $[0, 1]$ vs. unbounded BM25 scores) without requiring heuristic score calibration.

---

### `src/retrieval/reranker.py` ✅
**Promise:** Cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Computes joint query-document cross-attention for candidate chunks with graceful fallback to RRF ordering if the local model is unavailable.

---

### `src/generation/prompt_builder.py` ✅
**Promise:** Formats system and user prompts to enforce strict factual grounding. Enforces explicit refusal string (`"I cannot answer this question based on the provided document context."`) for ungrounded queries and requires inline citation tags `[Doc: <title>, Chunk: <index>]`.

---

### `src/generation/generator.py` ✅
**Promise:** Async chat completion client targeting Groq (`llama-3.1-8b-instant`) or OpenAI with non-blocking I/O and temperature 0.0 for factual accuracy.

---

### `src/pipeline/manual.py` ✅
**Promise:** Central orchestration function `run_manual_rag_pipeline(query)` executing parallel dense+sparse search, RRF fusion, cross-encoder reranking, prompt formatting, and LLM generation.

---

### `src/api/routes/query.py` ✅
**Promise:** FastAPI route `POST /query` validating requests via Pydantic `QueryRequest` and returning structured `QueryResponse` with answer, citations, and retrieval stage counts.

---

## Phase 3: Evaluation & Benchmarking Architecture ✅

### `evaluation/test_set.json`
50 hand-crafted test cases covering:
1. Direct Factual (14 items)
2. Multi-Chunk (10 items)
3. Conceptual / Synthesis (10 items)
4. Cross-Document (8 items)
5. Unanswerable / Out-of-Corpus (8 items)

### `src/evaluation/retrieval_eval.py`
Automated ablation engine evaluating:
- Config A: Dense only (`dense.py`)
- Config B: Sparse only (`sparse.py`)
- Config C: Hybrid RRF (`fusion.py`)
- Config D: Hybrid + Cross-Encoder (`reranker.py`)
Metrics: HitRate@1/3/5, Recall@1/3/5, MRR, NDCG@3/5.

### `src/evaluation/generation_eval.py`
Quality metrics evaluator assessing:
- Refusal accuracy on negative test cases
- Citation format precision & grounding
- Answer token-level F1 against ground truth

### `src/evaluation/latency_eval.py`
High-resolution timer calculating Mean, P50, P90, P95, and Max latencies for every individual stage and end-to-end execution.

---

## Infrastructure

### docker-compose.yml
- Service: `db` — `pgvector/pgvector:pg16` image. Postgres 16 with the `pgvector` extension pre-installed.
- Service: `app` — The FastAPI service. Depends on `db` health check.
- Volume: `pgdata` — Persistent Postgres data.

### Alembic Migrations
Location: `src/db/migrations/`. Not yet initialized. Must be run after ORM models are finalized.

---

## Key Engineering Decisions (Locked)

| Decision | What was chosen | Why |
|----------|-----------------|----|
| Vector Store | PostgreSQL + pgvector (with inmemory fallback) | pgvector forces manual RRF implementation; inmemory allows zero-dependency local development |
| Corpus format | Markdown text files (`corpus/raw/`) | Keeps focus on RAG retrieval mechanics rather than OCR preprocessing |
| Embedding provider | Cohere `embed-english-v3.0` (1024 dims) | Free developer tier; avoids local GPU requirement; production-realistic async API |
| Chunking strategy | Fixed-size with overlap (500/50) | Baseline approach; prevents boundary conceptual clipping |
| Hybrid Fusion | Reciprocal Rank Fusion ($k=60$) | Robust combination of lexical (BM25) and semantic (dense) ranks |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | High-precision second-stage reranker |
| LLM provider | Groq API (`llama-3.1-8b-instant`) | Fast inference and free tier |

---

## What Changes Should Trigger an Update to This File

- Any new file added to `src/`
- Any architectural decision made (storage, provider, algorithm)
- Any phase completed or newly started
- Any bug found that reveals a design flaw
- Any dependency added or removed from `pyproject.toml`

When updating: edit both the component description section and the Update Log in `AGENTS.md`.


