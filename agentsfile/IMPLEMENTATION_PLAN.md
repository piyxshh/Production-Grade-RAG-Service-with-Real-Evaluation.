# Implementation Plan
## Project 2: Production-Grade RAG Service

**Last Updated:** 2026-07-31  
**Status:** Pre-implementation (planning complete)

---

## Phase Overview

| Phase | Name | Est. Time | Core Learning |
|-------|------|-----------|---------------|
| 0 | Setup & Scaffolding | 1 day | Environment, Docker, Postgres+pgvector |
| 1 | Corpus & Ingestion | 2-3 days | Chunking strategies, embedding mechanics |
| 2 | Manual Retrieval Pipeline | 3-4 days | Dense search, BM25, RRF, reranking |
| 3 | Evaluation Suite | 3-4 days | RAGAS scoring, building a test set |
| 4 | Observability | 1-2 days | Langfuse/LangSmith tracing |
| 5 | LangChain/LangGraph Rebuild | 2-3 days | Framework abstraction comparison |
| 6 | Deployment | 2 days | Docker, hosting, prod config |

**Total: ~3-4 weeks (done properly, not rushed)**

---

## Architectural Decisions (Must Decide Before Phase 1)

These are decisions that affect the entire project structure. Decide each one explicitly — don't drift into a choice. Document your decision and the reason below.

### Decision 1: Corpus
- [ ] **Pick your corpus** — See PRD Section 3 for options
- [ ] Document it in README.md ("Corpus" section) with reasoning

### Decision 2: Vector Store
| Option | Pros | Cons | Best If |
|--------|------|------|---------|
| **pgvector** (Postgres) | One less service to run; you write RRF by hand | You write RRF by hand; no native BM25 | You want full control and to understand every line |
| **Qdrant** | Purpose-built; native sparse+dense hybrid | Abstracts some fusion logic | You want to move faster and learn the Qdrant API |

**Recommended:** pgvector — the whole point of Phase 1 is to write the hard parts yourself.

- [ ] **Make this decision** and document it in README.md

### Decision 3: Embedding Model
| Option | Cost | Speed | Control |
|--------|------|-------|---------|
| OpenAI `text-embedding-3-small` | ~$0.02/1M tokens | Fast (API) | Black box |
| `sentence-transformers` (local) | Free | Slower (local inference) | Full control |

- [ ] **Make this decision**

### Decision 4: LLM for Generation
| Option | Cost | Speed | Notes |
|--------|------|-------|-------|
| GPT-4o-mini | Very cheap | Fast | Safe default |
| Gemini 1.5 Flash | Very cheap / free tier | Fast | Good alternative |

- [ ] **Make this decision**

### Decision 5: Observability Tool
| Option | Self-hosted | Cost | UI Quality |
|--------|-------------|------|------------|
| Langfuse | Yes | Open source | Very good |
| LangSmith | No | Paid (generous free tier) | Excellent |

- [ ] **Make this decision**

---

## Phase 0: Setup & Scaffolding

> **Mode 3 territory — boilerplate is fine to generate.** You should still understand what each config file does.

### Tasks
- [ ] Create `pyproject.toml` with Poetry, add initial deps
- [ ] Create `Dockerfile` (Python 3.11, uvicorn)
- [ ] Create `docker-compose.yml` (Postgres with pgvector extension + app)
- [ ] Create `src/config.py` using Pydantic `BaseSettings`
- [ ] Create `src/main.py` FastAPI skeleton with `/health` route
- [ ] Create `.env.example` with all required env variable names
- [ ] Create `.gitignore`
- [ ] Initialize git repo, push to GitHub

### Key dependencies to add
```
fastapi uvicorn[standard] sqlalchemy asyncpg alembic
pgvector psycopg2-binary pydantic-settings python-dotenv
openai sentence-transformers rank-bm25
ragas langchain langchain-openai langgraph langfuse
httpx pytest pytest-asyncio
```

### Verify Phase 0 is done
- [ ] `docker-compose up -d` starts Postgres with pgvector extension enabled
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] Can connect to DB from app

---

## Phase 1: Corpus & Ingestion

> **You write `chunker.py` and `embedder.py` yourself. These are core learning pieces.**

### Step 1: Acquire & Explore Corpus
- [ ] Download/scrape your chosen corpus into `corpus/raw/`
- [ ] Open `notebooks/01_explore_corpus.ipynb` — count documents, look at length distribution, identify noise patterns
- [ ] Write down 3 things about your corpus that will make chunking hard

### Step 2: Write the Document Loader
- [ ] Create `src/ingestion/loaders.py`
- [ ] Handle your corpus format (PDF / HTML / markdown / text)
- [ ] Output: list of `{text: str, metadata: dict}` objects

### Step 3: Write the Chunker ⭐ (core learning piece)
- [ ] Create `src/ingestion/chunker.py`
- [ ] Implement **fixed-size chunking** with overlap first — understand the baseline
- [ ] Experiment in `notebooks/02_chunking_experiments.ipynb`: what chunk sizes produce meaningful units vs. too-split fragments?
- [ ] Be ready to answer: *What does a chunk promise to a retrieval system? Where does that promise break?*

### Step 4: Write the Embedder ⭐
- [ ] Create `src/ingestion/embedder.py`
- [ ] Implement `async embed_batch(texts: list[str]) -> list[list[float]]`
- [ ] Use `asyncio.gather` for concurrent embedding calls — don't call the API sequentially in a loop
- [ ] Handle rate limits and retries

### Step 5: DB Schema & Storage
- [ ] Create `src/db/models.py` — `Document` and `Chunk` tables
- [ ] Chunk table must store: `id`, `document_id`, `text`, `embedding vector(N)`, `metadata jsonb`
- [ ] Write Alembic migration: `alembic revision --autogenerate -m "initial"`, run it
- [ ] Create `src/ingestion/pipeline.py` — orchestrates loader → chunker → embedder → DB insert

### Step 6: Run the Ingestion
- [ ] Write `scripts/ingest_corpus.py` CLI script
- [ ] Run it, verify chunks + embeddings are in Postgres
- [ ] Answer: how long did ingestion take? How many chunks total?

### Verify Phase 1 is done
- [ ] `poetry run python scripts/ingest_corpus.py` completes without error
- [ ] Postgres contains your chunks with vector embeddings
- [ ] You can query the count from psql

---

## Phase 2: Manual Retrieval Pipeline

> **You write all four retrieval modules yourself. These are the core of the project.**

### Step 1: Dense Retrieval ⭐
- [ ] Create `src/retrieval/dense.py`
- [ ] Implement `async search(query_embedding: list[float], top_k: int) -> list[Chunk]`
- [ ] Use pgvector's `<=>` (cosine) or `<#>` (inner product) operator
- [ ] Return ranked list of chunks with scores
- [ ] Test in `notebooks/03_retrieval_experiments.ipynb`

### Step 2: Sparse Retrieval (BM25) ⭐
- [ ] Create `src/retrieval/sparse.py`
- [ ] Implement BM25 over your corpus using `rank_bm25` (load corpus into memory) or Postgres FTS
- [ ] Implement `search(query: str, top_k: int) -> list[Chunk]`
- [ ] Compare BM25 vs. dense results on 5 sample queries — where do they agree? Disagree?

### Step 3: Reciprocal Rank Fusion ⭐
- [ ] Create `src/retrieval/fusion.py`
- [ ] Implement RRF: `fuse(dense_results, sparse_results, k=60) -> list[Chunk]`
- [ ] Formula: `score(d) = Σ 1/(k + rank(d))` over each result list
- [ ] Be ready to explain: *What does RRF promise? Why `k=60`? What happens at `k=0`?*

### Step 4: Cross-Encoder Reranker ⭐
- [ ] Create `src/retrieval/reranker.py`
- [ ] Use `sentence-transformers` cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] Implement `rerank(query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]`
- [ ] Be ready to explain: *Why is reranking a separate stage? What does a bi-encoder (vector search) not do that a cross-encoder does?*

### Step 5: Prompt Assembly & Generation
- [ ] Create `src/generation/prompt_builder.py`
- [ ] Assemble a grounded prompt from top-k reranked chunks + the query
- [ ] Include chunk source metadata in the prompt so the LLM can cite it
- [ ] Create `src/generation/generator.py` — async call to LLM API

### Step 6: Wire the Manual Pipeline
- [ ] Create `src/pipeline/manual.py` — calls retrieval → RRF → rerank → prompt → generate
- [ ] Create `POST /query` route in `src/api/routes/query.py`
- [ ] Test end-to-end with 5 real questions

### Verify Phase 2 is done
- [ ] `POST /query` returns a grounded answer with source citations
- [ ] You can explain every step of what happened to produce that answer
- [ ] You've run the pipeline on at least 5 questions manually and inspected results

---

## Phase 3: Evaluation Suite

> **This is the highest-signal deliverable in the entire portfolio. Do not rush it.**

### Step 1: Write the Test Set ⭐
- [ ] Create `evaluation/test_set.json`
- [ ] Write **minimum 30 questions** yourself — do not generate them with an LLM
- [ ] Each entry: `{"question": "...", "ground_truth": "...", "relevant_chunk_ids": [...]}`
- [ ] Cover: factual questions, multi-hop reasoning questions, adversarial questions (things not in the corpus)

### Step 2: Baseline — Naive Vector-Only
- [ ] Create a simplified pipeline variant that skips BM25, RRF, and reranking
- [ ] Run RAGAS against the test set on this baseline
- [ ] Record scores in `evaluation/results/baseline_scores.json`

### Step 3: RAGAS Evaluation Harness ⭐
- [ ] Create `src/evaluation/ragas_eval.py`
- [ ] Implement: load test set → run pipeline for each question → collect `{question, answer, contexts, ground_truth}` → pass to RAGAS
- [ ] Output: scores for Faithfulness, Answer Relevance, Context Precision, Context Recall
- [ ] Write `scripts/run_eval.py` to execute this from CLI

### Step 4: Full Pipeline Evaluation
- [ ] Run RAGAS on the full hybrid+reranked pipeline
- [ ] Record scores in `evaluation/results/full_pipeline_scores.json`
- [ ] Fill in the before/after table in `README.md`

### Step 5: Find and Document a Failure Mode
- [ ] Identify at least one question the system handles badly
- [ ] Understand *why* it fails (wrong chunks retrieved? LLM hallucination despite good context?)
- [ ] Document it in the "Failure Mode" section of `README.md`

### Verify Phase 3 is done
- [ ] `poetry run python scripts/run_eval.py` runs and outputs scores
- [ ] Before/after table in README is filled with real numbers
- [ ] At least one failure mode is documented with root cause analysis

---

## Phase 4: Observability

- [ ] Choose between Langfuse and LangSmith (document your decision)
- [ ] Create `src/observability/tracing.py`
- [ ] Wrap each pipeline stage (embedding, dense retrieval, sparse retrieval, reranking, generation) with a trace span
- [ ] Run 20+ queries and capture cost/latency data
- [ ] Fill in the cost/latency table in `README.md`

---

## Phase 5: LangChain + LangGraph Rebuild

> **This phase is about comparison, not re-learning. Build it fast, then explain the diff.**

- [ ] Create `src/pipeline/langchain_pipeline.py` — replicate Phase 2 pipeline using LangChain
- [ ] Create `src/pipeline/agentic.py` — add retrieval-decision logic using LangGraph
  - Example: if top retrieved chunks have low relevance scores, trigger a second retrieval with a reformulated query
- [ ] Run RAGAS on the LangChain version — compare to manual pipeline scores
- [ ] Write the "LangChain vs Manual" section in `README.md`

---

## Phase 6: Deployment

- [ ] Choose deployment target (Railway, Render, AWS ECS, Fly.io, etc.)
- [ ] Ensure Postgres + pgvector is available in prod (e.g., Supabase, Neon, or a managed instance)
- [ ] Document the deployment URL in `README.md`
- [ ] Confirm `GET /health` returns 200 from the deployed URL

---

## Files the Mentor Will NOT Write For You

The following files are your core learning artifacts. They should be written by you first, reviewed together after:

| File | What It Teaches |
|------|-----------------|
| `src/ingestion/chunker.py` | Chunking strategies and their tradeoffs |
| `src/ingestion/embedder.py` | Async API calls, batching, rate limiting |
| `src/retrieval/dense.py` | pgvector query mechanics |
| `src/retrieval/sparse.py` | BM25 math and keyword retrieval |
| `src/retrieval/fusion.py` | RRF algorithm and fusion logic |
| `src/retrieval/reranker.py` | Cross-encoder vs bi-encoder distinction |
| `src/evaluation/ragas_eval.py` | Evaluation methodology, RAGAS metrics |
| `evaluation/test_set.json` | What makes a good eval test case |

---

## Files That Are Boilerplate (Fine to Scaffold Fully)

- `Dockerfile`, `docker-compose.yml`
- `pyproject.toml`
- `.env.example`, `.gitignore`
- `src/main.py` (FastAPI skeleton)
- `src/config.py` (Pydantic Settings)
- `src/db/session.py`, `src/db/migrations/`
- GitHub Actions CI YAML (if added)

---

## The Questions You Must Be Able to Answer Cold at the End

1. What is Reciprocal Rank Fusion? Why does the `k` hyperparameter exist?
2. What is the difference between a bi-encoder and a cross-encoder? Why can't you just use a cross-encoder from the start?
3. What does RAGAS "Faithfulness" actually measure? How is it computed?
4. What does async actually buy you in the embedding stage? What would happen if you made the calls sequentially?
5. What did LangChain abstract away in Phase 5? What did you lose visibility into?
6. Where does your hybrid retrieval pipeline fail? What type of query breaks it?
