"""One ingest run's outcome: a result per stage, as the run row stores it and the CLI prints it."""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.discover.models import DiscoverRunResult
from app.ingestion.embed.models import EmbedRunResult
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.models import StageRunResult
from app.ingestion.parse.models import ParseRunResult


class IngestRunResult(BaseModel):
    """Outcome of one ingest run: one result per stage."""

    STAGES: ClassVar[tuple[str, ...]] = ("discover", "fetch", "parse", "chunk", "embed")

    run_id: int
    corpus_version: str | None = None
    discover: DiscoverRunResult = Field(default_factory=DiscoverRunResult)
    fetch: FetchRunResult = Field(default_factory=FetchRunResult)
    parse: ParseRunResult = Field(default_factory=ParseRunResult)
    chunk: ChunkRunResult = Field(default_factory=ChunkRunResult)
    embed: EmbedRunResult = Field(default_factory=EmbedRunResult)

    @property
    def stages(self) -> dict[str, StageRunResult]:
        return {name: getattr(self, name) for name in self.STAGES}

    @property
    def unrecorded(self) -> frozenset[str]:
        """Stages that never reported, so a run cut short cannot close as a success."""
        return frozenset(self.STAGES) - self.model_fields_set

    def begin_document_stages(self) -> None:
        """Record the per-document stages as reporting before the loop that accumulates into them.

        unrecorded reads assignment, not value, and a loop where every document fails early
        accumulates into none of the later stages; without this they would read as never run.
        """
        self.fetch = FetchRunResult()
        self.parse = ParseRunResult()
        self.chunk = ChunkRunResult()

    @property
    def ok(self) -> bool:
        return not self.unrecorded and all(result.ok for result in self.stages.values())

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.COMPLETED if self.ok else IngestRunStatus.FAILED

    def report(self) -> dict[str, Any]:
        """The run as its row stores it: a report per stage, and null for one that never ran."""
        unrecorded = self.unrecorded
        return {
            name: None if name in unrecorded else result.report()
            for name, result in self.stages.items()
        }

    def summary(self) -> str:
        """The run as the CLI prints it: a line per stage, then the per-celex detail."""
        unrecorded = self.unrecorded
        return "\n".join(
            [
                f"run {self.run_id} ({self.corpus_version or 'not stamped'})",
                *(
                    f"  [{name}] {'not run' if name in unrecorded else result.summary()}"
                    for name, result in self.stages.items()
                ),
                *(
                    f"  {name} {line}"
                    for name, result in self.stages.items()
                    for line in result.details()
                ),
            ]
        )
