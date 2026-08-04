# Unit tests for the in-memory (mock) vector store.
# Verifies cosine-similarity ranking without any network or database.
import uuid

import pytest

from src.db.models import Chunk
from src.retrieval.stores import InMemoryVectorStore


def _store_with(items: list[tuple[str, list[float]]]) -> InMemoryVectorStore:
    """Build an InMemoryVectorStore directly with synthetic chunks (no I/O)."""
    store = InMemoryVectorStore()
    for text, vec in items:
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text=text,
            chunk_index=0,
            meta={},
            embedding=vec,
        )
        store._items.append((chunk, vec))
        store._norms.append(_norm(vec))
    store._loaded = True
    return store


def _norm(vec: list[float]) -> float:
    return sum(x * x for x in vec) ** 0.5


@pytest.mark.asyncio
async def test_returns_most_similar_first() -> None:
    store = _store_with(
        [
            ("python decorator syntax", [1.0, 0.0, 0.0]),
            ("react hooks rules", [0.0, 1.0, 0.0]),
            ("requests session cookies", [0.0, 0.0, 1.0]),
        ]
    )
    results = await store.search([1.0, 0.0, 0.0], top_k=2)
    texts = [chunk.text for chunk, _ in results]
    assert texts[0] == "python decorator syntax"
    assert texts[1] in {"react hooks rules", "requests session cookies"}


@pytest.mark.asyncio
async def test_top_k_limits_results() -> None:
    store = _store_with([(f"chunk-{i}", [1.0, float(i)]) for i in range(10)])
    results = await store.search([1.0, 0.0], top_k=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_similarity_is_one_for_identical_vector() -> None:
    store = _store_with([("same", [0.6, -0.8])])
    results = await store.search([0.6, -0.8], top_k=1)
    assert results[0][1] == pytest.approx(1.0)
