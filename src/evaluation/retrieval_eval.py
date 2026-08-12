"""Retrieval Evaluation Module: benchmark and compare retrieval configurations.

Configurations compared:
1. Dense retrieval only (Cohere semantic search)
2. Sparse retrieval only (BM25 Okapi)
3. Hybrid retrieval (Dense + BM25 fused with RRF k=60)
4. Full Hybrid + Reranker (Dense + BM25 + RRF + Cross-Encoder ms-marco-MiniLM-L-6-v2)

Metrics computed:
- HitRate@1, HitRate@3, HitRate@5
- Recall@1, Recall@3, Recall@5
- MRR (Mean Reciprocal Rank)
- NDCG@3, NDCG@5
"""
import asyncio
import math
import uuid
from typing import Any, Callable

from src.db.models import Chunk
from src.retrieval import dense, fusion, reranker, sparse


def compute_hit_rate(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Return 1.0 if at least one relevant chunk appears in top-k, else 0.0."""
    if not ground_truth_ids:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    return 1.0 if (top_k_ids & ground_truth_ids) else 0.0


def compute_recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Return fraction of ground-truth chunks retrieved in top-k."""
    if not ground_truth_ids:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    hits = len(top_k_ids & ground_truth_ids)
    return hits / len(ground_truth_ids)


def compute_reciprocal_rank(retrieved_ids: list[str], ground_truth_ids: set[str]) -> float:
    """Return 1 / rank of the first relevant chunk found (1-indexed), or 0.0."""
    if not ground_truth_ids:
        return 0.0
    for rank_idx, cid in enumerate(retrieved_ids, start=1):
        if cid in ground_truth_ids:
            return 1.0 / rank_idx
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    if not ground_truth_ids:
        return 0.0

    dcg = 0.0
    for rank_idx, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in ground_truth_ids:
            dcg += 1.0 / math.log2(rank_idx + 1)

    ideal_hits = min(k, len(ground_truth_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return (dcg / idcg) if idcg > 0.0 else 0.0


async def retrieve_dense_only(query: str, top_k: int = 10) -> list[Chunk]:
    results = await dense.search(query, top_k=top_k)
    return [item[0] for item in results]


async def retrieve_sparse_only(query: str, top_k: int = 10) -> list[Chunk]:
    results = await sparse.search(query, top_k=top_k)
    return [item[0] for item in results]


async def retrieve_hybrid_rrf(query: str, top_k: int = 10) -> list[Chunk]:
    d_task = dense.search(query, top_k=top_k * 2)
    s_task = sparse.search(query, top_k=top_k * 2)
    d_res, s_res = await asyncio.gather(d_task, s_task)
    fused = fusion.fuse_results(d_res, s_res, k=60, top_n=top_k)
    return [item[0] for item in fused]


async def retrieve_hybrid_reranked(query: str, top_k: int = 10) -> list[Chunk]:
    d_task = dense.search(query, top_k=top_k * 2)
    s_task = sparse.search(query, top_k=top_k * 2)
    d_res, s_res = await asyncio.gather(d_task, s_task)
    fused = fusion.fuse_results(d_res, s_res, k=60, top_n=top_k * 2)
    reranked = reranker.rerank_chunks(query, fused, top_n=top_k)
    return [item[0] for item in reranked]


RETRIEVAL_CONFIGS: dict[str, Callable[[str, int], Any]] = {
    "Dense Only": retrieve_dense_only,
    "BM25 Only": retrieve_sparse_only,
    "Hybrid (Dense+BM25+RRF)": retrieve_hybrid_rrf,
    "Hybrid + Cross-Encoder Reranker": retrieve_hybrid_reranked,
}


async def evaluate_retrieval_configuration(
    config_name: str,
    retriever_fn: Callable[[str, int], Any],
    test_set: list[dict],
    k_eval: int = 5,
) -> dict[str, Any]:
    """Evaluate a single retrieval pipeline configuration over the answerable test set."""
    answerable_items = [q for q in test_set if q.get("relevant_chunk_ids")]

    hit_rates_1: list[float] = []
    hit_rates_3: list[float] = []
    hit_rates_5: list[float] = []
    recalls_1: list[float] = []
    recalls_3: list[float] = []
    recalls_5: list[float] = []
    mrrs: list[float] = []
    ndcgs_3: list[float] = []
    ndcgs_5: list[float] = []

    per_query_details = []

    for item in answerable_items:
        query = item["question"]
        gt_ids = {str(uuid.UUID(cid)) for cid in item["relevant_chunk_ids"]}

        chunks = await retriever_fn(query, k_eval)
        retrieved_ids = [str(c.id) for c in chunks]

        hr1 = compute_hit_rate(retrieved_ids, gt_ids, k=1)
        hr3 = compute_hit_rate(retrieved_ids, gt_ids, k=3)
        hr5 = compute_hit_rate(retrieved_ids, gt_ids, k=5)

        rec1 = compute_recall_at_k(retrieved_ids, gt_ids, k=1)
        rec3 = compute_recall_at_k(retrieved_ids, gt_ids, k=3)
        rec5 = compute_recall_at_k(retrieved_ids, gt_ids, k=5)

        mrr = compute_reciprocal_rank(retrieved_ids, gt_ids)
        ndcg3 = compute_ndcg_at_k(retrieved_ids, gt_ids, k=3)
        ndcg5 = compute_ndcg_at_k(retrieved_ids, gt_ids, k=5)

        hit_rates_1.append(hr1)
        hit_rates_3.append(hr3)
        hit_rates_5.append(hr5)
        recalls_1.append(rec1)
        recalls_3.append(rec3)
        recalls_5.append(rec5)
        mrrs.append(mrr)
        ndcgs_3.append(ndcg3)
        ndcgs_5.append(ndcg5)

        per_query_details.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": query,
                "hit@1": hr1,
                "hit@3": hr3,
                "hit@5": hr5,
                "recall@5": rec5,
                "mrr": mrr,
                "ndcg@5": ndcg5,
                "retrieved_chunk_indices": [
                    c.chunk_index for c in chunks
                ],
            }
        )

    count = len(answerable_items)
    return {
        "config_name": config_name,
        "sample_size": count,
        "metrics": {
            "HitRate@1": sum(hit_rates_1) / count if count else 0.0,
            "HitRate@3": sum(hit_rates_3) / count if count else 0.0,
            "HitRate@5": sum(hit_rates_5) / count if count else 0.0,
            "Recall@1": sum(recalls_1) / count if count else 0.0,
            "Recall@3": sum(recalls_3) / count if count else 0.0,
            "Recall@5": sum(recalls_5) / count if count else 0.0,
            "MRR": sum(mrrs) / count if count else 0.0,
            "NDCG@3": sum(ndcgs_3) / count if count else 0.0,
            "NDCG@5": sum(ndcgs_5) / count if count else 0.0,
        },
        "query_details": per_query_details,
    }


async def run_all_retrieval_ablations(test_set: list[dict], k_eval: int = 5) -> dict[str, Any]:
    """Run all 4 retrieval configurations and collect ablation results."""
    results = {}
    for name, fn in RETRIEVAL_CONFIGS.items():
        results[name] = await evaluate_retrieval_configuration(name, fn, test_set, k_eval)
    return results
