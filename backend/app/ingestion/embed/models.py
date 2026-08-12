"""Embed-stage values: what one run's embedding pass changed."""

from pydantic import BaseModel, Field

from app.ingestion.exceptions import failure_reason


class EmbedOutcome(BaseModel):
    """How many chunk vectors a run filled in, how many were already there, and what would not."""

    embedded: int = 0
    already_embedded: int = 0
    """Already-vectored chunks corpus-wide, not just this run's documents."""
    failed: dict[str, str] = Field(default_factory=dict)

    def fail(self, celex: str, exc: Exception) -> None:
        """Record why a document's batch could not be embedded."""
        self.failed[celex] = failure_reason(exc)
