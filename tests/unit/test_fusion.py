"""Unit tests for the RRF fusion module."""
import uuid
import pytest
from src.db.models import Chunk
from src.retrieval.fusion import fuse_results


def _make_chunk(text: str, cid: uuid.UUID | None = None) -> Chunk:
    return Chunk(
        id=cid or uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        chunk_index=0,
        meta={"filename": "test.md"},
        embedding=None,
    )


def test_rrf_consensus_boost() -> None:
    """Document appearing high in BOTH dense and sparse lists should rank first."""
    chunk_a = _make_chunk("Chunk A (in both)")
    chunk_b = _make_chunk("Chunk B (dense only)")
    chunk_c = _make_chunk("Chunk C (sparse only)")

    # Dense: [B (rank 1), A (rank 2)]
    dense_results = [(chunk_b, 0.95), (chunk_a, 0.80)]
    # Sparse: [C (rank 1), A (rank 2)]
    sparse_results = [(chunk_c, 15.0), (chunk_a, 12.0)]

    # Scores with k=60:
    # A = 1/(60+2) + 1/(60+2) = 2/62 ≈ 0.032258
    # B = 1/(60+1) = 1/61 ≈ 0.016393
    # C = 1/(60+1) = 1/61 ≈ 0.016393
    fused = fuse_results(dense_results, sparse_results, k=60, top_n=3)

    assert len(fused) == 3
    assert fused[0][0].id == chunk_a.id
    assert fused[0][1] == pytest.approx(2.0 / 62.0)
    assert fused[1][1] == pytest.approx(1.0 / 61.0)
    assert fused[2][1] == pytest.approx(1.0 / 61.0)


def test_rrf_top_n_limits_output() -> None:
    """Output should strictly obey top_n parameter."""
    chunks = [_make_chunk(f"Chunk {i}") for i in range(10)]
    dense = [(c, float(10 - i)) for i, c in enumerate(chunks)]
    sparse = []

    fused = fuse_results(dense, sparse, k=60, top_n=4)
    assert len(fused) == 4


def test_rrf_invalid_k_raises() -> None:
    """k <= 0 should raise ValueError."""
    chunk = _make_chunk("Chunk")
    with pytest.raises(ValueError):
        fuse_results([(chunk, 0.9)], [], k=0)

