# Unit tests for BM25 sparse retrieval.
import pytest

from src.retrieval.sparse import tokenize


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_tokenize_ignores_punctuation_only() -> None:
    assert tokenize("...  ###  ???") == []


@pytest.mark.asyncio
async def test_empty_query_returns_empty() -> None:
    from src.retrieval.sparse import search

    assert await search("   ") == []
    assert await search("") == []


@pytest.mark.asyncio
async def test_search_returns_ranked_tuples() -> None:
    from src.retrieval.sparse import search

    results = await search("utility functions", top_k=3)
    assert 1 <= len(results) <= 3
    for chunk, score in results:
        assert score >= 0.0
        assert hasattr(chunk, "id") and hasattr(chunk, "text")