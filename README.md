# RAG Service — Production-Grade Retrieval-Augmented Generation

> **Portfolio Project 2.** A standalone, production-grade RAG service built to prove deep AI engineering understanding: hybrid retrieval, mathematical evaluation, observability, and agentic orchestration. Built twice — manually first, then with LangChain/LangGraph — so the abstraction gap is explicit and owned.

---

## What This Project Proves

| Gap It Closes | What This Demonstrates |
|---------------|------------------------|
| RAG beyond API calls | Manual chunking, embedding, RRF fusion, cross-encoder reranking |
| LLMOps maturity | RAGAS evaluation with before/after scores, not "looks right" vibes |
| Python/FastAPI depth | Genuine async service, not a script wrapped in Flask |
| Observability | Per-stage token, latency, and cost tracing via Langfuse |
| Framework understanding | LangChain rebuild after manual version — knows what the abstraction hides |

---

## Architecture

```
Client Request
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI Service                    │
│  POST /query                                         │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼──────────────┐
         │     Manual RAG Pipeline     │
         └──┬──────────────────────┬───┘
            │                      │
   ┌────────▼────────┐   ┌─────────▼────────┐
   │  Dense Retrieval │   │  Sparse Retrieval │
   │  (pgvector /     │   │  (BM25 / Postgres │
   │   cosine sim)    │   │   Full-Text)      │
   └────────┬────────┘   └─────────┬────────┘
            │                      │
            └──────────┬───────────┘
                       │
            ┌──────────▼──────────┐
            │  Reciprocal Rank    │
            │  Fusion (RRF)       │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Cross-Encoder      │
            │  Reranker           │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  Prompt Assembly    │
            │  + LLM Generation   │
            └──────────┬──────────┘
                       │
                   Answer + Sources
```

**Storage:**
- Postgres + pgvector — chunks, embeddings, metadata
- BM25 index — in-memory (rank_bm25) or Postgres FTS

**Infrastructure:**
- Docker + docker-compose (local)
- Deployed on [deployment target TBD]

---

## Corpus

**Corpus chosen:** [TBD — document your choice here and the reason]

**Why this corpus:**
[Write 1-2 sentences explaining your decision here after you make it]

**Volume:**
- Documents: [N]
- Chunks after processing: [N]
- Avg chunk size: [N tokens]

---

## Results (Fill In After Each Phase)

### Phase 1: RAGAS Evaluation — Naive vs. Hybrid+Reranked

| Metric | Naive Vector-Only | Hybrid + Reranked | Delta |
|--------|-------------------|-------------------|-------|
| Faithfulness | — | — | — |
| Answer Relevance | — | — | — |
| Context Precision | — | — | — |
| Context Recall | — | — | — |

> Scores above are on a 0.0–1.0 scale. Higher is better.

### Phase 2: Latency & Cost Per Pipeline Stage

| Stage | Avg Latency (ms) | Avg Cost (USD/req) | Tokens |
|-------|------------------|--------------------|--------|
| Embedding (query) | — | — | — |
| Dense Retrieval | — | — | — |
| BM25 Retrieval | — | — | — |
| RRF Fusion | — | — | — |
| Reranking | — | — | — |
| LLM Generation | — | — | — |
| **Total** | — | — | — |

---

## One Documented Failure Mode

> *Fill in after running the evaluation suite. Find a query the system handles badly, understand why, and document it here.*

**Query that fails:** [TBD]

**What the pipeline does:** [TBD — e.g., retrieves topically adjacent but factually wrong chunks]

**Why it fails:** [TBD — e.g., BM25 keyword match wins over semantic similarity here because the query uses generic terms]

**How it was handled (or why it wasn't fixed):** [TBD]

---

## Key Technical Decisions

### Chunking Strategy
[Document your chunking approach and why. What chunk size? Overlap? Semantic or fixed-size?]

### Hybrid Retrieval (RRF Fusion)
[Explain Reciprocal Rank Fusion in your own words. What does it promise? Where does it break?]

### Reranking
[Explain why reranking exists as a separate stage rather than just asking the vector DB for better results.]

### Rate-Limiting Algorithms Tradeoff
_(N/A — see Project 1)_

### LangChain vs Manual
[After Phase 2, document what LangChain abstracted away. What did you gain? What did you lose visibility into?]

---

## Project Structure

```
p2(RAG)/
├── README.md                      ← you are here
├── PRD.md                         ← full product requirements
├── IMPLEMENTATION_PLAN.md         ← phased build plan with task checklist
├── .env.example                   ← env variable template (no secrets)
├── .gitignore
├── docker-compose.yml             ← Postgres + pgvector + app
├── Dockerfile
├── pyproject.toml                 ← deps managed via Poetry
│
├── src/
│   ├── main.py                    ← FastAPI app entry point
│   ├── config.py                  ← Pydantic Settings from .env
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py           ← POST /query
│   │   │   ├── ingest.py          ← POST /ingest
│   │   │   └── health.py          ← GET /health
│   │   └── dependencies.py        ← FastAPI DI (DB sessions, pipeline)
│   │
│   ├── ingestion/
│   │   ├── chunker.py             ← [YOU WRITE] chunking logic
│   │   ├── embedder.py            ← [YOU WRITE] embedding API calls
│   │   ├── loaders.py             ← document loaders (PDF, HTML, text)
│   │   └── pipeline.py            ← orchestrates ingest end-to-end
│   │
│   ├── retrieval/
│   │   ├── dense.py               ← [YOU WRITE] pgvector similarity search
│   │   ├── sparse.py              ← [YOU WRITE] BM25 / Postgres FTS
│   │   ├── fusion.py              ← [YOU WRITE] Reciprocal Rank Fusion
│   │   └── reranker.py            ← [YOU WRITE] cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── prompt_builder.py      ← [YOU WRITE] manual prompt assembly
│   │   └── generator.py           ← LLM API call wrapper
│   │
│   ├── pipeline/
│   │   ├── manual.py              ← Phase 1: stitches all modules together
│   │   ├── langchain_pipeline.py  ← Phase 4: LangChain rebuild
│   │   └── agentic.py             ← Phase 4: LangGraph agentic version
│   │
│   ├── evaluation/
│   │   ├── ragas_eval.py          ← [YOU WRITE] RAGAS evaluation harness
│   │   └── metrics.py             ← helper metrics and result formatting
│   │
│   ├── observability/
│   │   └── tracing.py             ← Langfuse / LangSmith instrumentation
│   │
│   └── db/
│       ├── models.py              ← SQLAlchemy ORM models
│       ├── session.py             ← async DB session factory
│       └── migrations/            ← Alembic migration scripts
│
├── corpus/
│   ├── raw/                       ← downloaded source documents (gitignored if large)
│   └── processed/                 ← chunked + cleaned (gitignored if large)
│
├── evaluation/
│   ├── test_set.json              ← your 30+ hand-written Q&A pairs
│   └── results/                   ← RAGAS score outputs (JSON/CSV)
│
├── notebooks/
│   ├── 01_explore_corpus.ipynb    ← initial EDA of your corpus
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_retrieval_experiments.ipynb
│   └── 04_eval_analysis.ipynb
│
├── scripts/
│   ├── ingest_corpus.py           ← CLI: run full ingestion pipeline
│   └── run_eval.py                ← CLI: run evaluation suite
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_fusion.py
│   │   └── test_reranker.py
│   └── integration/
│       └── test_pipeline.py
│
└── docs/
    ├── architecture.md
    └── diagrams/
```

---

## Local Setup


### Prerequisites
- Python 3.11+
- Docker & docker-compose
- Poetry (`pip install poetry`)

### 1. Clone and configure
```bash
git clone <repo-url>
cd p2-rag
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Start infrastructure
```bash
docker-compose up -d
```

### 3. Install dependencies
```bash
poetry install
```

### 4. Run ingestion
```bash
poetry run python scripts/ingest_corpus.py
```

### 5. Start the API
```bash
poetry run uvicorn src.main:app --reload
```

### 6. Query it
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here", "mode": "manual"}'
```

---

## Evaluation

```bash
# Run full RAGAS evaluation suite
poetry run python scripts/run_eval.py

# Results will be written to evaluation/results/
```

---

