"""Dense retrieval: semantic (vector) search over the chunk corpus.

The query is embedded with the same Cohere model used at ingestion time, then
compared against every stored chunk embedding using cosine distance. The
closest ``top_k`` chunks are returned alongside their similarity scores.

Why cosine distance (``<=>``) and not inner product (``<#>``):
- Cosine distance measures the *angle* between two vectors and ignores their
  magnitude. Cohere embeddings are not guaranteed to be length-normalized, so
  inner product would silently bias results toward longer vectors.
- We use distance (ascending) to ORDER BY, then convert to a similarity score
  (``1 - distance``) so callers get a value where bigger = more relevant.

Why a plain exact scan here instead of an ANN index (IVFFlat/HNSW):
- The corpus is tiny (~50-500 chunks). Brute force is correct and fast at this
  scale. ANN indexes trade a small amount of recall for speed and only pay off
  at tens-of-thousands of vectors; building one now adds complexity for nothing.
- Index choice is documented in AGENTS.md as a Phase-2 open decision — revisit
  only if the corpus grows.

Edge cases handled:
- ``embedding IS NOT NULL`` — the Chunk.embedding column is nullable
  (models.py), so unembedded rows would otherwise be silently dropped by the
  ``<=>`` comparison. We filter them explicitly.
- Empty/whitespace query returns an empty list before hitting the API/DB.
"""
from sqlalchemy import cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from pgvector.sqlalchemy import Vector

from src.db.models import Chunk
from src.db.session import get_session
from src.ingestion.embedder import embed_batch

DEFAULT_TOP_K = 5


async def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    session: AsyncSession | None = None,
) -> list[tuple[Chunk, float]]:
    """Return the ``top_k`` chunks most similar to ``query``.

    Each result is a ``(Chunk, similarity)`` pair where ``similarity`` is in
    ``[0, 1]`` with 1.0 meaning an identical direction (most similar).

    ``session`` is optional: pass one in to reuse an existing FastAPI DB
    dependency session; otherwise a fresh session is created and closed here.
    """
    if not query or not query.strip():
        return []

    query_vector = (await embed_batch([query], input_type="search_query"))[0]

    query_vec_literal = cast(query_vector, Vector)
    distance = Chunk.embedding.cosine_distance(query_vec_literal)

    stmt = (
        select(Chunk, distance.label("distance"))
        .where(Chunk.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(top_k)
    )

    owns_session = session is None
    if owns_session:
        session = get_session()

    try:
        rows = (await session.execute(stmt)).all()
    finally:
        if owns_session:
            await session.close()

    return [(row[0], 1.0 - float(row[1])) for row in rows]
