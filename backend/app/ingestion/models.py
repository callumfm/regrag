"""Ingestion domain values: run update body and the composed run report."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.ingestion.chunk.models import ChunkDelta
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchDelta
from app.ingestion.parse.models import ParseDelta
from app.ingestion.stage import IngestStageDelta


class IngestRunUpdate(BaseModel):
    """Partial update body for an ingest run."""

    status: IngestRunStatus | None = None
    corpus_version: str | None = None
    completed_at: datetime | None = None


class RunReport(BaseModel):
    """Outcome of one ingest run: one delta per stage."""

    run_id: int
    corpus_version: str | None = None
    fetch: FetchDelta = Field(default_factory=FetchDelta)
    parse: ParseDelta = Field(default_factory=ParseDelta)
    chunk: ChunkDelta = Field(default_factory=ChunkDelta)

    @property
    def stages(self) -> dict[str, IngestStageDelta]:
        return {name: value for name, value in self if isinstance(value, IngestStageDelta)}

    @property
    def ok(self) -> bool:
        return all(delta.ok for delta in self.stages.values())

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.COMPLETED if self.ok else IngestRunStatus.FAILED

    def summary(self) -> str:
        """The run as the CLI prints it: a line per stage, then the per-ref detail."""
        return "\n".join(
            [
                f"run {self.run_id} ({self.corpus_version or 'not stamped'})",
                *(f"  [{name}] {delta.summary()}" for name, delta in self.stages.items()),
                *(
                    f"  {name} {line}"
                    for name, delta in self.stages.items()
                    for line in delta.details()
                ),
            ]
        )
