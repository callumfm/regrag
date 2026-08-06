"""Ingestion domain values describing a run as a whole."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.parse.models import ParseRunResult
from app.ingestion.stage import StageRunResult


class IngestRunUpdate(BaseModel):
    """Partial update body for an ingest run."""

    status: IngestRunStatus | None = None
    corpus_version: str | None = None
    completed_at: datetime | None = None


class IngestRunResult(BaseModel):
    """Outcome of one ingest run: one result per stage."""

    run_id: int
    corpus_version: str | None = None
    fetch: FetchRunResult = Field(default_factory=FetchRunResult)
    parse: ParseRunResult = Field(default_factory=ParseRunResult)
    chunk: ChunkRunResult = Field(default_factory=ChunkRunResult)

    @property
    def stages(self) -> dict[str, StageRunResult]:
        return {"fetch": self.fetch, "parse": self.parse, "chunk": self.chunk}

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.stages.values())

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.COMPLETED if self.ok else IngestRunStatus.FAILED

    def summary(self) -> str:
        """The run as the CLI prints it: a line per stage, then the per-ref detail."""
        return "\n".join(
            [
                f"run {self.run_id} ({self.corpus_version or 'not stamped'})",
                *(f"  [{name}] {result.summary()}" for name, result in self.stages.items()),
                *(
                    f"  {name} {line}"
                    for name, result in self.stages.items()
                    for line in result.details()
                ),
            ]
        )
