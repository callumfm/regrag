"""Persisted chunks: one row per retrievable unit of a regulation."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Computed, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import config
from app.core.db.schema import BaseSchema
from app.ingestion.enums import SectionKind
from app.ingestion.schemas import IngestRun

SEARCH_VECTOR_SQL = (
    """setweight(to_tsvector('english', citation || ' ' || coalesce(title, '')), 'A')"""
    """ || setweight(to_tsvector('english', "text"), 'B')"""
)


class DocumentChunk(BaseSchema):
    """One retrievable unit of a regulation, content-addressed so it survives re-runs."""

    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("celex", "content_hash", "occurrence"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id", ondelete="CASCADE"))

    celex: Mapped[str]
    topic: Mapped[str]
    content_hash: Mapped[str]
    occurrence: Mapped[int]

    kind: Mapped[SectionKind]
    article: Mapped[str | None]
    annex: Mapped[str | None]
    title: Mapped[str | None]
    paragraph: Mapped[str | None]
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(String))
    citation: Mapped[str]

    part: Mapped[int]
    parts: Mapped[int]

    text: Mapped[str]
    references: Mapped[list[dict]] = mapped_column(JSONB)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(config.EMBED_DIMENSIONS))
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed(SEARCH_VECTOR_SQL, persisted=True)
    )

    run: Mapped[IngestRun] = relationship()
