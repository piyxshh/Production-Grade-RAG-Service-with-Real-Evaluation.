"""Unit tests for the reranker module."""
import uuid
import pytest
from src.db.models import Chunk
from src.retrieval.reranker import rerank_chunks


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        chunk_index=0,
        meta={"filename": "test.md"},
        embedding=None,
    )


def test_reranker_empty_input() -> None:
    assert rerank_chunks("", []) == []
    assert rerank_chunks("test query", []) == []
    chunk = _make_chunk("test chunk")
    assert rerank_chunks("   ", [chunk]) == []


def test_reranker_returns_ranked_chunks() -> None:
    chunks = [
        _make_chunk("A complete guide to Python decorators and syntax."),
        _make_chunk("React useEffect hook dependency array guide."),
        _make_chunk("Configuring session retry headers in requests."),
    ]
    query = "How to write Python decorators?"
    reranked = rerank_chunks(query, chunks, top_n=2)
    assert len(reranked) == 2
    assert isinstance(reranked[0][0], Chunk)
    assert isinstance(reranked[0][1], float)

