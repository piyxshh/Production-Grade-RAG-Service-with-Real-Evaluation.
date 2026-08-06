"""Reciprocal Rank Fusion (RRF): merge dense and sparse result lists into one.

Formula:
    score(d) = sum over each list of 1 / (k + rank(d))
where rank(d) is the 1-indexed position of document d in that list.

Why RRF:
- Dense retrieval returns cosine similarities bounded in [0, 1].
- BM25 returns unbounded, corpus-dependent positive float scores.
- Because these scores are on entirely incomparable distributions, direct score averaging
  or linear combination requires careful calibration.
- RRF sidesteps calibration by operating solely on *rank positions*.

Why k=60:
- The constant k dampens the advantage of top ranks.
- At k=0, rank 1 receives score 1.0 while rank 2 receives 0.5 (a 50% drop).
- At k=60, rank 1 receives 1/61 (~0.01639) and rank 2 receives 1/62 (~0.01613), allowing
  consensus across multiple lists to outscore an isolated top hit.
"""
import logging
import uuid
from src.config import settings
from src.db.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_K = 60
DEFAULT_TOP_N = 10


def fuse_results(
    dense_results: list[tuple[Chunk, float]],
    sparse_results: list[tuple[Chunk, float]],
    k: int = DEFAULT_K,
    top_n: int = DEFAULT_TOP_N,
) -> list[tuple[Chunk, float]]:
    """Merge ranked lists from dense and sparse retrieval using Reciprocal Rank Fusion.

    Args:
        dense_results: Ordered list of (Chunk, cosine_similarity) from dense retrieval.
        sparse_results: Ordered list of (Chunk, bm25_score) from sparse retrieval.
        k: Smoothing constant (default: 60).
        top_n: Number of fused results to return.

    Returns:
        List of (Chunk, rrf_score) sorted descending by rrf_score.
    """
    if k <= 0:
        raise ValueError(f"Smoothing constant k must be positive, got {k}")

    rrf_scores: dict[uuid.UUID, float] = {}
    chunk_lookup: dict[uuid.UUID, Chunk] = {}

    # Process dense results
    for rank_idx, (chunk, _) in enumerate(dense_results, start=1):
        cid = chunk.id if isinstance(chunk.id, uuid.UUID) else uuid.UUID(str(chunk.id))
        chunk_lookup[cid] = chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank_idx))

    # Process sparse BM25 results
    for rank_idx, (chunk, _) in enumerate(sparse_results, start=1):
        cid = chunk.id if isinstance(chunk.id, uuid.UUID) else uuid.UUID(str(chunk.id))
        chunk_lookup[cid] = chunk
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank_idx))

    # Sort descending by fused RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused = [(chunk_lookup[cid], rrf_scores[cid]) for cid in sorted_ids]
    return fused[:top_n]

