"""Sparse retrieval: BM25 (Okapi) keyword search over the chunk corpus.

BM25 scores how well a query's *literal tokens* match each chunk's tokens,
weighted by:
- term frequency (TF): how often the term appears in a chunk
- inverse document frequency (IDF): how *rare* the term is across the corpus —
  a word in few documents is more discriminating, so it carries more weight
- document-length normalization: longer chunks get diluted so a long doc isn't
  auto-favored just by having more words.

It is a purely lexical retriever — it matches exact tokens, not meaning.
Consequently it excels at queries with distinctive/technical/rare keywords and
fails at paraphrases and synonyms. That is the exact *opposite* failure mode of
dense retrieval, which is why hybrid retrieval (RRF, fusion.py) combines both:
each compensates for the other's blind spot.

Why in memory / shared records:
- BM25 needs the corpus tokenized in memory. We reuse the same record source as
  the mock vector store (src/retrieval/stores.py) so the *same chunks, with the
  same ids*, are indexed here. Fusion can then merge dense + sparse result lists
  by chunk id without id-mismatches.
"""
import logging
import re

from rank_bm25 import BM25Okapi

from src.db.models import Chunk
from src.retrieval.stores import _chunk_from_record, load_chunk_records

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into alphanumeric tokens."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


_index: BM25Okapi | None = None
_chunks: list[Chunk] = []


async def _ensure_index() -> tuple[BM25Okapi, list[Chunk]]:
    """Build (once) the in-memory BM25 index over the same chunks as dense."""
    global _index, _chunks
    if _index is None:
        records = await load_chunk_records()
        corpus = [tokenize(r["text"]) for r in records]
        _chunks = [_chunk_from_record(r) for r in records]
        _index = BM25Okapi(corpus)
        logger.info("BM25 index built over %d chunks", len(_chunks))
    return _index, _chunks


async def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[Chunk, float]]:
    """Return the ``top_k`` chunks with the highest BM25 score for ``query``.

    Each result is a ``(Chunk, score)`` pair ordered best first. Higher score
    means a stronger literal keyword match to the query.
    """
    if not query or not query.strip():
        return []

    bm25, chunks = await _ensure_index()
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(chunks[i], float(scores[i])) for i in ranked[:top_k]]