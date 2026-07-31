# RAG Service — Production-Grade Retrieval-Augmented Generation

A standalone, production-grade RAG service featuring hybrid retrieval, mathematical evaluation, observability, and agentic orchestration.

---

## Features

- **Hybrid Retrieval**: Combines dense vector search (pgvector) with sparse keyword matching (BM25) via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Re-ranks fused results using a cross-encoder model for high precision.
- **Async Architecture**: Fully asynchronous FastAPI backend for concurrent embedding and retrieval calls.
- **Evaluation Suite**: Built-in RAGAS evaluation harness to measure Faithfulness, Answer Relevance, and Context Precision/Recall.
- **Observability**: Integrated tracing for token usage, latency, and costs per pipeline stage.
- **Agentic Orchestration**: Uses LangGraph to make retrieval decisions (e.g., re-retrieval on low confidence).

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
         │        RAG Pipeline         │
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

---

## Project Structure

```
p2(RAG)/
├── README.md                      
├── PRD.md                         
├── IMPLEMENTATION_PLAN.md         
├── .env.example                   
├── .gitignore
├── docker-compose.yml             
├── Dockerfile
├── pyproject.toml                 
│
├── src/
│   ├── main.py                    
│   ├── config.py                  
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py           
│   │   │   ├── ingest.py          
│   │   │   └── health.py          
│   │   └── dependencies.py        
│   │
│   ├── ingestion/
│   │   ├── chunker.py             
│   │   ├── embedder.py            
│   │   ├── loaders.py             
│   │   └── pipeline.py            
│   │
│   ├── retrieval/
│   │   ├── dense.py               
│   │   ├── sparse.py              
│   │   ├── fusion.py              
│   │   └── reranker.py            
│   │
│   ├── generation/
│   │   ├── prompt_builder.py      
│   │   └── generator.py           
│   │
│   ├── pipeline/
│   │   ├── manual.py              
│   │   ├── langchain_pipeline.py  
│   │   └── agentic.py             
│   │
│   ├── evaluation/
│   │   ├── ragas_eval.py          
│   │   └── metrics.py             
│   │
│   ├── observability/
│   │   └── tracing.py             
│   │
│   └── db/
│       ├── models.py              
│       ├── session.py             
│       └── migrations/            
│
├── corpus/
│   ├── raw/                       
│   └── processed/                 
│
├── evaluation/
│   ├── test_set.json              
│   └── results/                   
│
├── notebooks/
│   ├── 01_explore_corpus.ipynb    
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_retrieval_experiments.ipynb
│   └── 04_eval_analysis.ipynb
│
├── scripts/
│   ├── ingest_corpus.py           
│   └── run_eval.py                
│
├── tests/
│   ├── unit/
│   └── integration/
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
