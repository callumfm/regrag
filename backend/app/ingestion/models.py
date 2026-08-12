"""Ingestion values every stage shares."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.ingestion.enums import IngestRunStatus


class IngestRunUpdate(BaseModel):
    """Partial update body for an ingest run."""

    status: IngestRunStatus | None = None
    corpus_version: str | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
