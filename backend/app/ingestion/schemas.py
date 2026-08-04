"""Ingest tracking: one row per corpus fetch run, one per document fetched in it."""

from datetime import datetime

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.schema import BaseSchema
from app.ingestion.enums import IngestRunStatus


class IngestRun(BaseSchema):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[IngestRunStatus]
    corpus_version: Mapped[str | None]
    completed_at: Mapped[datetime | None]

    documents: Mapped[list["IngestedDocument"]] = relationship(back_populates="run")


class IngestedDocument(BaseSchema):
    __tablename__ = "ingested_documents"
    __table_args__ = (UniqueConstraint("ingest_run_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id", ondelete="CASCADE"))
    name: Mapped[str]
    source: Mapped[str]
    ref: Mapped[str]
    resolved_ref: Mapped[str]
    topic: Mapped[str]
    url: Mapped[str]
    sha256: Mapped[str]
    size_bytes: Mapped[int]
    fetched_at: Mapped[datetime]

    run: Mapped[IngestRun] = relationship(back_populates="documents")
