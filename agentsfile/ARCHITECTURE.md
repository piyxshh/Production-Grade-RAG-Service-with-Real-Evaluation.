# ARCHITECTURE.md — System Architecture
## Production-Grade RAG Service

**Last Updated:** 2026-08-04  
**Current Phase:** Phase 2 — Retrieval Pipeline (in progress)

---

## Overview

This service answers natural-language questions by:
1. **Ingesting** a corpus of documents — loading, chunking, embedding, and storing them in Postgres.
2. **Retrieving** the most relevant chunks for a given question using hybrid search (dense vector + BM25 keyword).
3. **Reranking** those results with a cross-encoder for precision.
4. **Generating** a grounded answer using an LLM, citing only the retrieved context.

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
│                              │
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
 PHASE 2: RETRIEVAL PIPELINE (runs per query, online)
════════════════════════════════════════════════════════════════

  User Question (string)
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
│ 2. <=> cosine│          │    keyword   │
│    distance  │          │    search    │
│    in pgvec- │          │    (rank_    │
│    tor       │          │    bm25)     │
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
         │  list            │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Reranker        │  reranker.py
         │                  │
         │  Cross-encoder   │
         │  model reads     │
         │  (query, chunk)  │
         │  pairs directly  │
         │  and re-scores   │
         │  for precision   │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Prompt Builder  │  generation/prompt_builder.py
         │                  │
         │  Assembles:      │
         │  - System prompt │
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
         │  (Llama 3 model) │
         └────────┬─────────┘
                  │
                  ▼
        Answer + Source Citations
```

---

## Component Descriptions

### `src/ingestion/loaders.py`
**Promise:** Takes chaotic files from `corpus/raw/` and returns a uniform `list[dict]` structure.  
**Output:** `[{"text": "...", "metadata": {"filename": "...", "source_url": "..."}}]`  
**Handles:** `.md`, `.txt`, `.pdf` (text-native only).  
**Known edge case:** Scanned image PDFs return empty strings. OCR path was explicitly rejected — swap the corpus file instead.

---

### `src/ingestion/chunker.py`
**Promise:** Breaks large documents into small, overlapping, searchable units without severing concepts.  
**Two functions:**
- `chunk_text(text, chunk_size=500, overlap=50) -> list[str]` — Core split logic. Step = chunk_size - overlap.
- `chunk_documents(documents) -> list[dict]` — Wraps `chunk_text`, attaches original metadata + `chunk_index` to every chunk.  

**The overlap guarantee:** A sentence spanning the 500-char boundary appears at the end of Chunk N and the beginning of Chunk N+1 (overlap=50 chars). No concept is entirely severed.

**Where it breaks:** Character-based chunking splits mid-word or mid-sentence. Semantic chunking (splitting on paragraph/sentence boundaries) would be more precise but requires an NLP library and is out of scope for Phase 1.

---

### `src/ingestion/embedder.py`
**Promise:** Takes a list of text strings, returns a list of 1024-dimensional float vectors (one per text), in the same order, reliably even under API rate limits.  
**API:** Cohere `embed-english-v3.0` via `AsyncClientV2`.  
**Mechanisms:**
- **Batching:** Groups texts into batches of 96 (Cohere's request limit).
- **Concurrency:** `asyncio.gather` fans out all batches simultaneously.
- **Backpressure:** `asyncio.Semaphore(10)` prevents more than 10 requests in-flight at once.
- **Retry:** Tenacity catches `TooManyRequestsError`, `ServiceUnavailableError`, `InternalServerError`; retries up to 5× with exponential backoff + random jitter.

**Where it breaks:** The Semaphore is per-call-site, not global. If two concurrent callers both embed independently, effective concurrency doubles to 20. This is acceptable at current corpus scale.

---

### `src/db/models.py`
**Two tables:**

#### `Document`
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key, generated by Postgres |
| source_url | String(2048) | Stable external key used for deduplication on re-ingest |
| title | String(512) | Optional |
| metadata | JSONB | Arbitrary metadata from loader |
| created_at | DateTime | Server-set timestamp |

#### `Chunk`
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| document_id | UUID FK → Document | CASCADE delete |
| text | Text | Raw chunk string |
| chunk_index | Integer | Position within parent document |
| embedding | Vector(1024) | pgvector type; 1024 dims = Cohere model output |
| metadata | JSONB | Carries filename, chunk_index, any loader metadata |
| created_at | DateTime | Server-set timestamp |

**Relationship:** `Document.chunks` → `Chunk` (one-to-many). Deleting a Document cascades to all its Chunks.

---

### `src/ingestion/pipeline.py`
**Promise:** Orchestrates the full ingest from files on disk to rows in Postgres, with no data loss from partial failures.  
**Flow:**
1. `asyncio.to_thread(load_documents)` — offloads synchronous file I/O from the event loop.
2. `chunk_documents` — splits all loaded docs.
3. `embed_batch([c["text"] for c in chunks])` — embeds all chunks in one concurrent pass.
4. Per-document loop: `zip(doc_chunks, doc_vectors)` pairs text with vector, saves `Chunk` rows.
5. Per-document `commit()` / `rollback()` — one bad document cannot abort the whole corpus.  

**Deduplication:** Checks `source_url` before inserting a `Document`. Re-ingesting the same file updates metadata instead of creating duplicates.

---

### `src/retrieval/dense.py` — NOT YET IMPLEMENTED
**Planned:** Async function that embeds the user's query, then performs a `pgvector` `<=>` cosine distance query to retrieve the top-K most semantically similar `Chunk` rows.

---

### `src/retrieval/sparse.py` — NOT YET IMPLEMENTED
**Planned:** BM25 keyword search using `rank_bm25` (in-memory). Loads all chunk texts from Postgres at startup, builds a `BM25Okapi` index, and queries it against the tokenized user query.

---

### `src/retrieval/fusion.py` — NOT YET IMPLEMENTED
**Planned:** Reciprocal Rank Fusion (RRF).  
**Formula:** `score(d) = Σ 1 / (k + rank(d))` where k=60 is the standard dampening constant.  
**Purpose:** Merges the ranked lists from dense and sparse retrieval into a single unified ranking without requiring the two lists' scores to be on the same scale.

---

### `src/retrieval/reranker.py` — NOT YET IMPLEMENTED
**Planned:** Cross-encoder reranker. Unlike bi-encoders (used in dense retrieval), a cross-encoder reads the query and each chunk together as a single input, enabling far more accurate relevance scoring at the cost of speed. Applied only to the top-N fused results, not the full corpus.

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

| Decision | What was considered | What was chosen | Why |
|----------|--------------------|-----------------|----|
| Vector Store | Qdrant vs pgvector | pgvector (Postgres) | Qdrant abstracts RRF internally; pgvector forces manual implementation which teaches the algorithm |
| Corpus format | Scanned PDFs (rejected), Markdown | Markdown text files | OCR path was considered and explicitly rejected to keep focus on RAG, not document processing |
| Embedding provider | OpenAI (paid), sentence-transformers (local GPU), Cohere (free API) | Cohere | Free developer tier; avoids local GPU requirement; production-realistic async API |
| Chunking strategy | Semantic, recursive, fixed-size | Fixed-size with overlap (500/50) | Baseline approach; highest learning clarity; semantic chunking added complexity without additional learning value at this stage |
| LLM provider | OpenAI (paid), Groq (free) | Groq | Free tier; avoids API costs for portfolio project |

---

## What Changes Should Trigger an Update to This File

- Any new file added to `src/`
- Any architectural decision made (storage, provider, algorithm)
- Any phase completed or newly started
- Any bug found that reveals a design flaw
- Any dependency added or removed from `pyproject.toml`

When updating: edit both the component description section and the Update Log in `AGENTS.md`.
