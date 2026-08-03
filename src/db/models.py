# SQLAlchemy ORM models.
# Document + Chunk tables backed by Postgres + pgvector.
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from pgvector.sqlalchemy import Vector

# Embedding dimension of the configured Cohere model (embed-english-v3.0).
EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    """Declarative base every ORM model inherits from."""


class Document(Base):
    """A single source document pulled from the raw corpus."""

    __tablename__ = "document"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=func.gen_random_uuid()
    )
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    meta: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    """A chunk of a document plus its embedding vector."""

    __tablename__ = "chunk"

    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=func.gen_random_uuid()
    )
    document_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    meta: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")