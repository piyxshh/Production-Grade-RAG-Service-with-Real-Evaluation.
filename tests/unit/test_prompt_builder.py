"""Unit tests for prompt_builder module."""
import uuid
import pytest
from src.db.models import Chunk
from src.generation.prompt_builder import build_rag_prompt, REFUSAL_MESSAGE


def _make_chunk(text: str, filename: str = "doc.md", cidx: int = 0) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        text=text,
        chunk_index=cidx,
        meta={"filename": filename, "chunk_index": cidx},
        embedding=None,
    )


def test_build_rag_prompt_structure() -> None:
    chunk1 = _make_chunk("Python decorators wrap functions.", filename="decorators.md", cidx=1)
    chunk2 = _make_chunk("React hooks allow functional state.", filename="react.md", cidx=2)

    sys_prompt, user_prompt = build_rag_prompt("How do decorators work?", [chunk1, chunk2])

    assert REFUSAL_MESSAGE in sys_prompt
    assert "decorators.md" in user_prompt
    assert "react.md" in user_prompt
    assert "How do decorators work?" in user_prompt
    assert "--- CONTEXT BLOCK 1 ---" in user_prompt
    assert "--- CONTEXT BLOCK 2 ---" in user_prompt


def test_build_rag_prompt_empty_chunks() -> None:
    sys_prompt, user_prompt = build_rag_prompt("Any question?", [])
    assert "No relevant context found." in user_prompt
    assert "Any question?" in user_prompt
