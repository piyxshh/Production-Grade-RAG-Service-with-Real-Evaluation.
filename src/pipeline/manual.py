"""Manual RAG Pipeline: end-to-end orchestration without framework magic.

Pipeline Data Flow:
1. User Query
2. Parallel Retrieval: Dense (Cosine Similarity) + Sparse (BM25) via asyncio.gather
3. Fusion: Reciprocal Rank Fusion (RRF) with k=60 to merge non-comparable score distributions
4. Reranker: Cross-Encoder (ms-marco-MiniLM-L-6-v2) for fine-grained token-level cross attention
5. Grounded Prompt Builder: Inject top chunks with document and chunk identifiers
6. LLM Generator: Groq / OpenAI chat completion for grounded answer generation
7. Result Delivery: Grounded answer + traceable source citations
"""
import asyncio
import logging
from typing import Any

from src.config import settings
from src.generation import generator, prompt_builder
from src.retrieval import dense, fusion, reranker, sparse

logger = logging.getLogger(__name__)


async def run_manual_rag_pipeline(
    query: str,
    *,
    top_k_retrieval: int = 5,
    top_n_rerank: int = 3,
    session: Any = None,
) -> dict[str, Any]:
    """Execute the end-to-end manual RAG pipeline for a given query.

    Args:
        query: Natural language question.
        top_k_retrieval: Candidates to fetch from each retriever (dense & sparse).
        top_n_rerank: Number of top chunks to retain after cross-encoder reranking.
        session: Optional SQLAlchemy AsyncSession for DB-backed stores.

    Returns:
        Dictionary containing question, answer, sources, and retrieval statistics.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return {
            "question": query,
            "answer": "Please provide a valid question.",
            "sources": [],
            "retrieval_stats": {
                "dense_count": 0,
                "sparse_count": 0,
                "fused_count": 0,
                "reranked_count": 0,
            },
        }

    # Step 1: Parallel dense and sparse retrieval
    dense_task = dense.search(clean_query, top_k=top_k_retrieval, session=session)
    sparse_task = sparse.search(clean_query, top_k=top_k_retrieval)
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    logger.debug(
        "Retrieved %d dense and %d sparse candidates for query: %s",
        len(dense_results),
        len(sparse_results),
        clean_query,
    )

    # Step 2: Reciprocal Rank Fusion
    fused_results = fusion.fuse_results(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=settings.rrf_k,
        top_n=top_k_retrieval * 2,
    )

    # Step 3: Cross-Encoder Reranking
    reranked_results = reranker.rerank_chunks(
        query=clean_query,
        candidates=fused_results,
        top_n=top_n_rerank,
    )

    selected_chunks = [item[0] for item in reranked_results]

    # Format source citation metadata
    sources = []
    for chunk in selected_chunks:
        meta = chunk.meta or {}
        filename = meta.get("filename") or "Unknown"
        title = meta.get("title") or filename
        cidx = chunk.chunk_index if chunk.chunk_index is not None else meta.get("chunk_index", 0)
        snippet = chunk.text.strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        sources.append(
            {
                "document_id": str(chunk.document_id) if chunk.document_id else None,
                "chunk_id": str(chunk.id) if chunk.id else None,
                "title": title,
                "filename": filename,
                "chunk_index": cidx,
                "snippet": snippet,
            }
        )

    # Step 4: Prompt Construction
    system_prompt, user_prompt = prompt_builder.build_rag_prompt(
        query=clean_query,
        candidates=selected_chunks,
    )

    # Step 5: LLM Generation
    answer = await generator.generate_answer(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return {
        "question": clean_query,
        "answer": answer,
        "sources": sources,
        "retrieval_stats": {
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused_results),
            "reranked_count": len(reranked_results),
        },
    }

