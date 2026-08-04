# Product Requirements Document (PRD)
## Project 2: Production-Grade RAG Service with Real Evaluation

**Author:** Piyush  
**Status:** Active  
**Last Updated:** 2026-07-31  
**Target Completion:** ~3-4 weeks from start

---

## 1. Background & Motivation

### 1.1 The Gap This Closes
The existing resume demonstrates fast delivery of AI-powered product features, but cannot demonstrate:
- Deep understanding of how RAG pipelines actually work under the hood
- Any experience building or evaluating a pipeline mathematically (not just "does it look right?")
- Production-grade observability, evaluation methodology, or LLMOps maturity
- Python/FastAPI as a serious async backend

The best existing agentic work (Advera) is private. This project becomes the public, evidenced replacement.

### 1.2 The Deeper Engineering Goal
Building this project by hand first — before any framework — installs a critical reflex that matters beyond the resume:

> For any component you build, ask:
> 1. What does it promise?
> 2. What mechanism actually keeps that promise?
> 3. Where does the mechanism break?
> 4. What do you do about the break?

This reflex, not the specific tech, is the differentiator. The project is the material to practice it on.

---

## 2. What We Are Building

A **standalone, production-grade RAG (Retrieval-Augmented Generation) service** with the following characteristics:
- Ingests a real, messy corpus of documents
- Retrieves relevant context using **hybrid search** (dense vectors + BM25 keyword matching)
- Reranks retrieved results using a cross-encoder
- Generates grounded answers using an LLM
- Evaluates its own output quality using a mathematical evaluation harness (RAGAS)
- Is fully instrumented for cost, latency, and token observability
- Is built **twice**: once manually, once with LangChain/LangGraph — so the abstraction gap is explicit and understood

This is not a chatbot wrapper. It is an AI system with a verifiable quality bar.

---

## 3. Corpus Decision

The corpus needs to be:
- **Real volume**: at minimum a few hundred documents, ideally 1000+
- **Messy**: inconsistent formatting, varying quality, duplicate info across documents
- **Meaningful**: a domain where factual grounding matters and hallucinations are detectable

**Candidates (to be decided before Phase 1 starts):**
| Option | Source | Volume | Messiness | Notes |
|--------|--------|--------|-----------|-------|
| Python docs | docs.python.org | ~500 pages | Medium | Well-known, good for testing retrieval quality |
| FastAPI / Pydantic docs | Combined | ~300 pages | Medium-High | Topically narrow, overlapping content |
| ArXiv papers (ML) | arXiv API | Thousands | High | PDFs are messy, chunking is hard — high learning value |
| Wikipedia (specific domain) | Wikipedia API | Unlimited | High | Real-world messiness, freely available |

> **Decision point:** Pick one before writing any ingestion code. Document the choice and why in the README.

---

## 4. Functional Requirements

### 4.1 Phase 1 — Manual Pipeline (Core Learning)

| ID | Requirement | Priority |
|----|-------------|----------|
| F-01 | Ingest documents from the chosen corpus into Postgres (pgvector) | P0 |
| F-02 | Chunk documents with a configurable strategy (fixed-size, semantic, recursive) | P0 |
| F-03 | Generate dense embeddings using an embedding model API | P0 |
| F-04 | Store chunks, embeddings, and raw text in Postgres | P0 |
| F-05 | Implement BM25 index over the corpus (in-memory or via Postgres FTS) | P0 |
| F-06 | Implement vector similarity search (cosine/inner product) via pgvector | P0 |
| F-07 | Implement Reciprocal Rank Fusion (RRF) to merge dense + sparse results | P0 |
| F-08 | Apply a cross-encoder reranker to the fused results | P0 |
| F-09 | Assemble a grounded prompt manually from top-k reranked chunks | P0 |
| F-10 | Call an LLM API and return the generated answer | P0 |
| F-11 | Expose all of the above as a FastAPI REST endpoint | P0 |

### 4.2 Phase 2 — Evaluation Suite

| ID | Requirement | Priority |
|----|-------------|----------|
| E-01 | Write a minimum 30-question hand-crafted test set (Q, expected answer, source chunk) | P0 |
| E-02 | Score the pipeline on Faithfulness using RAGAS | P0 |
| E-03 | Score the pipeline on Answer Relevance using RAGAS | P0 |
| E-04 | Score the pipeline on Context Precision and Recall using RAGAS | P0 |
| E-05 | Run eval on naive vector-only retrieval AND hybrid+reranked — report both numbers | P0 |
| E-06 | Document at least one failure mode found during evaluation | P0 |

### 4.3 Phase 3 — Observability

| ID | Requirement | Priority |
|----|-------------|----------|
| O-01 | Instrument pipeline with Langfuse or LangSmith | P0 |
| O-02 | Trace: token usage per stage (embedding, retrieval, generation) | P0 |
| O-03 | Trace: latency per stage | P0 |
| O-04 | Trace: cost per request | P0 |

### 4.4 Phase 4 — Framework Rebuild (LangChain + LangGraph)

| ID | Requirement | Priority |
|----|-------------|----------|
| LC-01 | Rebuild the manual pipeline using LangChain | P1 |
| LC-02 | Add agentic retrieval logic using LangGraph (e.g., re-retrieve if low confidence) | P1 |
| LC-03 | Document the specific abstractions LangChain replaced and what was lost/gained | P1 |

---

## 5. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NF-01 | FastAPI endpoints must be fully async (no blocking I/O in routes) |
| NF-02 | The service must be containerized with Docker and docker-compose |
| NF-03 | All secrets in `.env`, never hardcoded |
| NF-04 | Core logic must be covered by unit tests |
| NF-05 | README must include real benchmark numbers (RAGAS scores, latency, cost) before project is "done" |
| NF-06 | Deployed somewhere real — not just "runs on my laptop" |

---

## 6. Definition of Done

A project phase is "done" when you can, without notes, explain:
1. What it promises
2. What mechanism keeps that promise
3. Where the promise breaks
4. What you did about the break

...for that specific phase's core technical decision.

For the full project, "done" means:
- RAGAS before/after numbers are in the README
- Cost/latency table from observability is in the README
- One documented failure mode is in the README
- The code is deployed and accessible

---

## 7. Out of Scope

- A user-facing UI (this is a backend service, not a product)
- Authentication / rate limiting (that's Project 1's domain)
- Fine-tuning any model
- Any LangGraph agent that takes external actions — retrieval decisions only

---

## 8. Open Questions (Decide Before Coding)

1. **Which corpus?** See Section 3.
2. **pgvector or Qdrant?** See implementation plan for tradeoffs.
3. **Which embedding model?** OpenAI `text-embedding-3-small` (cost-efficient) vs. a local model via `sentence-transformers` (zero API cost, full control)?
4. **Which LLM for generation?** GPT-4o-mini (cheap, fast) vs. Gemini Flash (generous free tier)?
5. **Which observability tool?** Langfuse (self-hostable, open source) vs. LangSmith (managed, better UI)?

---

*This PRD is a living document. Update it when architectural decisions are made.*
