"""Prompt assembly: format retrieved context chunks and query into a strictly grounded prompt.

Key Grounding Guarantees:
1. Context Isolation: The system prompt instructs the model to operate strictly within
   the provided context and refrain from using ungrounded external assumptions.
2. Refusal Baseline: If the provided documents do not contain the answer, the model is
   instructed to explicitly refuse with a standardized message rather than hallucinating.
3. Traceable Citations: Metadata (filename/title and chunk index) is formatted alongside
   each chunk text so the model can cite exact sources inline.
"""
from typing import Sequence, Union
from src.db.models import Chunk

REFUSAL_MESSAGE = "I cannot answer this question based on the provided document context."

SYSTEM_PROMPT = """You are a precision QA assistant. Your task is to answer the user's question using ONLY the facts directly provided in the context below.

STRICT OPERATIONAL RULES:
1. Answer ONLY based on the provided context chunks. Do NOT bring in external knowledge, assumptions, or speculations.
2. If the provided context does not contain sufficient information to answer the question truthfully and completely, respond EXACTLY with:
"{refusal}"
3. Whenever stating a fact or code snippet derived from a chunk, you MUST cite the source inline using the format:
[Doc: <filename_or_title>, Chunk: <chunk_index>]
4. Keep the answer clear, concise, and technically accurate.
""".strip().format(refusal=REFUSAL_MESSAGE)


def format_context_chunk(chunk: Chunk, index: int) -> str:
    """Format an individual chunk with its metadata header for prompt injection."""
    meta = chunk.meta or {}
    filename = meta.get("filename") or "Unknown"
    title = meta.get("title") or filename
    chunk_index = chunk.chunk_index if chunk.chunk_index is not None else meta.get("chunk_index", index)
    chunk_id = str(chunk.id) if chunk.id else f"chunk-{index}"

    return (
        f"--- CONTEXT BLOCK {index + 1} ---\n"
        f"Document: {title}\n"
        f"Chunk Index: {chunk_index}\n"
        f"Chunk ID: {chunk_id}\n"
        f"Content:\n{chunk.text.strip()}\n"
    )


def build_rag_prompt(
    query: str,
    candidates: Sequence[Union[Chunk, tuple[Chunk, float]]],
) -> tuple[str, str]:
    """Assemble the system prompt and user prompt for RAG generation.

    Args:
        query: The user's question.
        candidates: Sequence of Chunk objects or (Chunk, score) tuples.

    Returns:
        tuple[str, str]: (system_prompt, user_prompt)
    """
    chunks: list[Chunk] = [c[0] if isinstance(c, tuple) else c for c in candidates]

    if not chunks:
        context_str = "No relevant context found."
    else:
        context_str = "\n".join(
            format_context_chunk(chunk, idx) for idx, chunk in enumerate(chunks)
        )

    user_prompt = (
        f"CONTEXT DOCUMENTS:\n"
        f"==================\n"
        f"{context_str}\n"
        f"==================\n\n"
        f"USER QUESTION: {query.strip()}\n\n"
        f"GROUNDED ANSWER:"
    )

    return SYSTEM_PROMPT, user_prompt

