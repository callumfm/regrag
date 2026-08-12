"""Embed-stage values: what one run's embedding pass changed."""

from pydantic import BaseModel, Field

from app.ingestion.exceptions import failure_reason


class EmbedFailure(BaseModel):
    """Chunks one document could not embed, and the first error that stopped them."""

    chunks: int = 0
    reason: str = ""

    def describe(self) -> str:
        unit = "chunk" if self.chunks == 1 else "chunks"
        return f"{self.chunks} {unit}: {self.reason}"


class EmbedOutcome(BaseModel):
    """How many chunk vectors a run filled in, how many were already there, and what would not."""

    embedded: int = 0
    already_embedded: int = 0
    """Already-vectored chunks corpus-wide, not just this run's documents."""
    failed: dict[str, EmbedFailure] = Field(default_factory=dict)

    def fail(self, celex: str, exc: Exception, *, chunks: int) -> None:
        """Record a failed batch against its document, accumulating across batches."""
        failure = self.failed.setdefault(celex, EmbedFailure(reason=failure_reason(exc)))
        failure.chunks += chunks

    @property
    def failures(self) -> dict[str, str]:
        """Each failed document's loss on one line: how many chunks are missing, and why."""
        return {celex: failure.describe() for celex, failure in self.failed.items()}
