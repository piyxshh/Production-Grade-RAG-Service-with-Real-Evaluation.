"""Dense retrieval: semantic (vector) search over the chunk corpus.

The query is embedded with the same Cohere model used at ingestion time, then
the resulting vector is handed to the configured ``VectorStore`` backend
(``src/retrieval/stores.py``), which ranks chunks by cosine similarity and
returns the closest ``top_k`` with their scores.

Why retrieval talks to a store interface instead of Postgres directly:
- Today there is no database available on this machine (Docker/pgvector is a
  later phase), so the in-memory store lets the whole retrieval pipeline run
  and be tested without any infra.
- Later, swapping to the real pgvector backend is a one-line config change
  (``VECTOR_STORE=pgvector``). Nothing else in this module changes.
- The embedding of the query itself is deliberately *not* part of the store:
  embedding is model logic (Cohere), storage is where vectors live. Keeping
  them separate means either backend can be used interchangeably.

Edge cases handled:
- Empty/whitespace query returns an empty list before touching the API/store.
- The store filters rows with missing embeddings (see stores.py); cosine is
  undefined without both vectors present.
"""
from src.db.models import Chunk
from src.ingestion.embedder import embed_batch
from src.retrieval.stores import get_vector_store

DEFAULT_TOP_K = 5


async def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    session=None,
) -> list[tuple[Chunk, float]]:
    """Return the ``top_k`` chunks most similar to ``query``.

    Each result is a ``(Chunk, similarity)`` pair where ``similarity`` is in
    ``[0, 1]`` with 1.0 meaning an identical direction (most similar).

    ``session`` is forwarded to database-backed stores (e.g. a FastAPI
    ``get_db`` dependency); the in-memory store ignores it.
    """
    if not query or not query.strip():
        return []

    query_vector = (await embed_batch([query], input_type="search_query"))[0]

    store = get_vector_store()
    return await store.search(query_vector, top_k=top_k, session=session)
