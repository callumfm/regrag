"""Ingest tracking: one row per corpus fetch run."""

from datetime import datetime

from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.schema import BaseSchema
from app.ingestion.enums import IngestRunStatus


class IngestRun(BaseSchema):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[IngestRunStatus]
    corpus_version: Mapped[str | None]
    completed_at: Mapped[datetime | None]
