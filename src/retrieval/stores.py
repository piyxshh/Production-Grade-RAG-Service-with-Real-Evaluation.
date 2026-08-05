"""Vector store abstraction: the seam between retrieval and storage.

Why this exists
---------------
The project will eventually run against Postgres + pgvector inside Docker
(the ``docker-compose.yml`` path). Right now Docker isn't installed on this
machine, so retrieval must run against an in-memory store that needs no
database at all. Instead of writing retrieval code twice, both backends
implement the same ``VectorStore`` contract and a factory selects one from
configuration (``VECTOR_STORE=inmemory|pgvector``).

Swapping later is a one-line config change:
    VECTOR_STORE=pgvector   # after Docker/Postgres is up and corpus ingested

Both stores return ``(Chunk, similarity)`` pairs with similarity in ``[0, 1]``
so downstream stages (fusion, reranker, prompt builder) are identical no matter
which backend produced the results.
"""
import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from math import sqrt
from pathlib import Path

from sqlalchemy import cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from pgvector.sqlalchemy import Vector

from src.config import settings
from src.db.models import Chunk
from src.db.session import get_session
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_batch
from src.ingestion.loaders import load_documents

logger = logging.getLogger(__name__)

# Snapshot file for the in-memory store (built by scripts/build_mock_index.py
# or generated on first load from corpus/raw/).
MOCK_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "mock_index.json"

# Stable namespace so chunk ids are deterministic across runs.
DOC_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _norm(vec: list[float]) -> float:
    return sqrt(sum(x * x for x in vec))


class VectorStore(ABC):
    """Contract every retrieval backend implements."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        session: AsyncSession | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return ``(Chunk, similarity)`` pairs ranked best first.

        ``session`` is only used by database-backed stores; in-memory stores
        ignore it. Kept on the interface so callers can pass a FastAPI
        ``get_db`` session regardless of backend.
        """


class InMemoryVectorStore(VectorStore):
    """Mock backend: cosine similarity over chunks held in memory.

    Needs no database, no pgvector, no Docker. Chunks and their embeddings come
    from ``data/mock_index.json`` (see scripts/build_mock_index.py). If the
    snapshot is missing the store builds it once from ``corpus/raw/`` using the
    exact same loader -> chunker -> embedder pipeline the real ingestion uses —
    only the *storage* step differs, which is the point of the abstraction.
    """

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []
        self._norms: list[float] = []
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        records = await load_chunk_records()
        for r in records:
            chunk = _chunk_from_record(r)
            vec: list[float] = r["embedding"]
            self._items.append((chunk, vec))
            self._norms.append(_norm(vec))
        self._loaded = True
        logger.info("Mock vector store loaded %d chunks", len(self._items))

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        session: AsyncSession | None = None,
    ) -> list[tuple[Chunk, float]]:
        await self._ensure_loaded()
        query_norm = _norm(query_vector)
        scored: list[tuple[Chunk, float]] = []
        for (chunk, vec), norm in zip(self._items, self._norms):
            dot = sum(a * b for a, b in zip(query_vector, vec))
            sim = dot / (query_norm * norm) if query_norm and norm else 0.0
            scored.append((chunk, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]


class PgVectorStore(VectorStore):
    """Real backend: pgvector cosine-distance search in Postgres.

    This is the Docker/Postgres path. Requires the ``vector`` extension and an
    ingested ``chunk`` table (see src/ingestion/pipeline.py). Reuses the same
    SQL semantics as the in-memory store: cosine distance ascending, filtered
    to rows whose embedding is present, limit top_k.
    """

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        *,
        session: AsyncSession | None = None,
    ) -> list[tuple[Chunk, float]]:
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


async def _load_records() -> list[dict]:
    """Return the mock index records, building them if the snapshot is missing."""
    if MOCK_INDEX_PATH.is_file():
        with MOCK_INDEX_PATH.open(encoding="utf-8") as fh:
            records = json.load(fh)
        logger.info("Loaded mock index from %s", MOCK_INDEX_PATH)
        return records

    logger.info("No mock index snapshot; building from corpus/raw/")
    documents = await asyncio.to_thread(load_documents)
    chunks = chunk_documents(documents)
    vectors = await embed_batch([c["text"] for c in chunks])

    records = []
    for chunk, vec in zip(chunks, vectors):
        filename = chunk["metadata"].get("filename", "unknown")
        cidx = chunk["metadata"].get("chunk_index", 0)
        records.append(
            {
                "id": str(uuid.uuid5(DOC_NS, f"{filename}#{cidx}")),
                "document_id": str(uuid.uuid5(DOC_NS, filename)),
                "text": chunk["text"],
                "chunk_index": cidx,
                "metadata": chunk["metadata"],
                "embedding": vec,
            }
        )
    return records


def _chunk_from_record(r: dict) -> Chunk:
    """Build a Chunk from a mock-index record (shared by all stores)."""
    return Chunk(
        id=uuid.UUID(r["id"]),
        document_id=uuid.UUID(r["document_id"]),
        text=r["text"],
        chunk_index=r["chunk_index"],
        meta=r["metadata"],
        embedding=r["embedding"],
    )


async def load_chunk_records() -> list[dict]:
    """Public accessor so sparse retrieval can index the same chunk texts.

    Reusing the same record source guarantees dense and sparse (and later RRF
    fusion) operate on chunks with identical ``id`` values — the requirement
    for merging result lists by chunk.
    """
    return await _load_records()


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the configured store, built once (lazy singleton)."""
    global _store
    if _store is None:
        if settings.vector_store == "pgvector":
            _store = PgVectorStore()
        else:
            _store = InMemoryVectorStore()
        logger.info("Using vector store: %s", settings.vector_store)
    return _store
