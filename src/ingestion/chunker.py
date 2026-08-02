"""
Chunking logic: take a document's text and split it into chunks.

Fixed-size chunking with overlap — the baseline every more sophisticated
strategy is compared against.
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split a single string into fixed-size chunks with `overlap` chars shared.

    Raises ValueError if chunk_size is not positive or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text or ""
    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    end = len(text)

    while start < end:
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Chunk the output of the loader, keeping each document's metadata.

    Each chunk is a dict: ``{"text": str, "metadata": {...source, "chunk_index": i}}``.
    """
    chunks: list[dict] = []
    for doc in documents:
        metadata = doc.get("metadata", {})
        for index, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append(
                {
                    "text": chunk,
                    "metadata": {**metadata, "chunk_index": index},
                }
            )
    return chunks
