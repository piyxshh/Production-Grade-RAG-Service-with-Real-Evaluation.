"""Cross-encoder reranker: high-precision relevance scoring over candidate chunks.

Architectural Difference:
- Bi-encoders (used in dense retrieval) encode query and document into separate vectors
  independently: f(query) and g(document). Similarity is a simple dot product.
  This allows pre-computing and indexing millions of vectors offline for O(1) retrieval.
- Cross-encoders encode (query, document) pairs *together* through all transformer
  self-attention layers: f(query, document). This captures rich, token-level cross-attention
  between the question and context, yielding substantially higher relevance accuracy.

Why Reranking is a Second Stage:
- Cross-encoders cannot be pre-indexed and require a full transformer forward pass for every
  candidate at query time (O(N) latency).
- Running a cross-encoder across a 100,000-chunk corpus is computationally prohibitive.
- Solution: Retrieve broad candidates (top 10-20) via fast hybrid search (dense + BM25),
  then rerank to precise top 3-5 with the cross-encoder.
"""
import logging
from typing import Sequence, Union

from src.config import settings
from src.db.models import Chunk

logger = logging.getLogger(__name__)

_cross_encoder = None
_model_load_failed = False


def _get_cross_encoder():
    """Lazy-load the CrossEncoder model singleton."""
    global _cross_encoder, _model_load_failed
    if _cross_encoder is None and not _model_load_failed:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model: %s", settings.cross_encoder_model)
            _cross_encoder = CrossEncoder(settings.cross_encoder_model)
        except Exception as err:
            logger.warning(
                "Failed to load cross-encoder model '%s' (%s). Falling back to original retrieval order.",
                settings.cross_encoder_model,
                err,
            )
            _model_load_failed = True
    return _cross_encoder


def is_cross_encoder_loaded() -> bool:
    """Return True if a neural CrossEncoder model is loaded and active."""
    return _get_cross_encoder() is not None



def rerank_chunks(
    query: str,
    candidates: Sequence[Union[Chunk, tuple[Chunk, float]]],
    top_n: int = 5,
) -> list[tuple[Chunk, float]]:
    """Rerank candidate chunks using a cross-encoder model.

    Args:
        query: The user's search query.
        candidates: List of Chunk objects or (Chunk, score) tuples.
        top_n: Maximum number of reranked chunks to return.

    Returns:
        List of (Chunk, cross_encoder_score) sorted descending.
    """
    if not candidates or not query.strip():
        return []

    # Normalize candidates to list of Chunk objects
    chunks: list[Chunk] = [c[0] if isinstance(c, tuple) else c for c in candidates]

    model = _get_cross_encoder()
    if model is None:
        # Fallback: maintain candidate order with normalized mock/original scores
        logger.debug("Cross-encoder unavailable; passing through top %d candidates.", top_n)
        return [(chunk, 1.0 / (idx + 1)) for idx, chunk in enumerate(chunks[:top_n])]

    pairs = [[query, chunk.text] for chunk in chunks]
    try:
        scores = model.predict(pairs)
        ranked = sorted(
            zip(chunks, [float(s) for s in scores]),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_n]
    except Exception as err:
        logger.error("Cross-encoder prediction error: %s. Falling back to candidate order.", err)
        return [(chunk, 1.0 / (idx + 1)) for idx, chunk in enumerate(chunks[:top_n])]

