"""Build the in-memory vector store snapshot (data/mock_index.json).

The mock store (VECTOR_STORE=inmemory) can either build its index lazily from
corpus/raw/ on first query, or load a pre-built snapshot. This script builds
the snapshot once so repeated runs don't re-call the embedding API.

Usage:
    poetry run python scripts/build_mock_index.py

The snapshot uses the exact same loader -> chunker -> embedder pipeline as the
real ingestion (src/ingestion/pipeline.py); only the storage step differs.
"""
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

# Make `src` importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.chunker import chunk_documents  # noqa: E402
from src.ingestion.embedder import embed_batch  # noqa: E402
from src.ingestion.loaders import load_documents  # noqa: E402

logging.basicConfig(level=logging.INFO)

DOC_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "mock_index.json"


async def main() -> None:
    documents = load_documents()
    logging.info("Loaded %d documents", len(documents))
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

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(records), encoding="utf-8")
    logging.info("Wrote %d chunks to %s", len(records), OUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
