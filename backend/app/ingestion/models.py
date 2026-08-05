"""Pydantic models for the ingestion domain."""

from datetime import datetime

from pydantic import BaseModel

from app.ingestion.enums import IngestRunStatus


class IngestRunUpdate(BaseModel):
    """Partial update body for an ingest run."""

    status: IngestRunStatus | None = None
    corpus_version: str | None = None
    completed_at: datetime | None = None


class ChunkDelta(BaseModel):
    """What reconciling one document's chunks changed."""

    added: int = 0
    removed: int = 0
    unchanged: int = 0
