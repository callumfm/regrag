"""Persisted chunks: one row per retrievable unit of a regulation."""

from sqlalchemy import ARRAY, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.schema import BaseSchema
from app.ingestion.enums import SectionKind


class DocumentChunk(BaseSchema):
    """One retrievable unit of a regulation, content-addressed so it survives re-runs."""

    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("ref", "content_hash", "occurrence"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ref: Mapped[str]
    topic: Mapped[str]
    content_hash: Mapped[str]
    occurrence: Mapped[int]
    kind: Mapped[SectionKind]
    article: Mapped[str | None]
    annex: Mapped[str | None]
    title: Mapped[str | None]
    paragraph: Mapped[str | None]
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(String))
    part: Mapped[int]
    parts: Mapped[int]
    citation: Mapped[str]
    text: Mapped[str]
    references: Mapped[list[dict]] = mapped_column(JSONB)
    corpus_version: Mapped[str | None]
