"""
Ingestion orchestrator: loader -> document row -> chunker -> embedder -> chunk rows.

The pipeline is fully async and fault-tolerant:
  - Synchronous file IO (the loader) is offloaded to a worker thread so the event
    loop is never blocked reading the local corpus.
  - Embedding calls are batched and concurrent inside ``embed_batch``.
  - Each document's row is persisted first (flush assigns its UUID), its chunks +
    vectors are then inserted, and everything commits together per document. A
    failure in one document is caught and logged so it never fails the whole corpus.
"""
import asyncio
import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Chunk, Document
from src.db.session import get_session
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_batch
from src.ingestion.loaders import load_documents

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Summary of a single ingestion run."""

    documents_ingested: int = 0
    chunks_ingested: int = 0
    skipped_documents: list[str] = field(default_factory=list)


async def _load(corpus_dir: str | None) -> list[dict]:
    """Run the synchronous loader off the event loop."""
    return await asyncio.to_thread(load_documents, corpus_dir)


async def _persist_document(
    session: AsyncSession, doc: dict
) -> Document:
    """Persist (or dedupe) one loader dict, returning (and flushing) its row.

    ``metadata["filename"]`` is treated as the stable external key and stored in
    ``source_url``. Re-ingesting the same corpus updates metadata in place instead
    of creating duplicate documents; the flushed row guarantees ``id`` is set so
    chunks can link to it.
    """
    metadata = doc.get("metadata") or {}
    source = metadata.get("filename") or ""
    title = metadata.get("title") or source or None
    source_url = metadata.get("source_url") or source or None

    if source_url:
        existing = await session.scalar(
            select(Document).where(Document.source_url == source_url).limit(1)
        )
        if existing is not None:
            logger.info("Document already present, skipping insert: %s", source_url)
            existing.meta = metadata
            return existing

    doc_row = Document(title=title, source_url=source_url, meta=metadata)
    session.add(doc_row)
    await session.flush()  # assigns doc_row.id
    return doc_row


async def ingest_corpus(
    corpus_dir: str | None = None,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    session: AsyncSession | None = None,
) -> IngestResult:
    """Run the full ingestion pipeline over ``corpus_dir``.

    Orchestration order:
      1. Loader returns ``{"text", "metadata"}`` dicts from the raw corpus.
      2. Each loader output dict becomes a persisted ``Document`` row.
      3. Loader output is passed to the chunker for text splitting.
      4. Chunked texts go to the embedder to produce vectors.
      5. Chunks and vectors are zipped together and saved as ``Chunk`` rows.

    ``chunk_documents`` emits chunks in source order, and ``embed_batch`` preserves
    that order, so *all* chunks are embedded in one concurrent pass and then sliced
    back per document by a running offset. When ``session`` is omitted a fresh one is
    created and every document commits independently, so a partial failure never
    leaves the corpus half-written.
    """
    owned_session = session is None
    session = session or get_session()

    result = IngestResult()
    try:
        documents = await _load(corpus_dir)
        logger.info("Loaded %d documents from corpus", len(documents))

        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        vectors = await embed_batch([c["text"] for c in chunks])

        offset = 0
        for doc in documents:
            doc_chunks = chunk_subset_by_source(
                chunks, doc.get("metadata") or {}
            )
            # embed_batch order == chunk order, so slice by this doc's chunk count.
            count = len(doc_chunks)
            doc_vectors = vectors[offset : offset + count]
            offset += count
            try:
                record = await _persist_document(session, doc)
                await insert_chunks(session, record, doc_chunks, doc_vectors)
                result.documents_ingested += 1
                result.chunks_ingested += count
                if owned_session:
                    await session.commit()
            except Exception as exc:  # keep the pipeline moving
                name = (doc.get("metadata") or {}).get("filename", "<unknown>")
                logger.error("Failed to ingest %s: %s", name, exc)
                result.skipped_documents.append(name)
                await session.rollback()
    finally:
        if owned_session:
            await session.close()

    return result


def chunk_subset_by_source(chunks: list[dict], metadata: dict) -> list[dict]:
    """Return the slices of ``chunks`` belonging to ``metadata``'s filename."""
    source = metadata.get("filename")
    if source is None:
        return chunks
    return [c for c in chunks if c["metadata"].get("filename") == source]


async def insert_chunks(
    session: AsyncSession,
    document: Document,
    chunks: list[dict],
    vectors: list[list[float]],
) -> int:
    """Zip ``chunks`` with aligned ``vectors`` and persist chunk rows. Returns count."""
    if not chunks:
        return 0

    rows = [
        Chunk(
            document_id=document.id,
            text=chunk["text"],
            chunk_index=chunk["metadata"].get("chunk_index", idx),
            embedding=vector,
            meta=chunk["metadata"],
        )
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    session.add_all(rows)
    await session.flush()
    return len(rows)