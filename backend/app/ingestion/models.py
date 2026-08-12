"""Ingestion values every stage shares: one run's outcome, and the row it closes out."""

from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.core.models import FrozenModel
from app.ingestion.chunk.models import ChunkCounts
from app.ingestion.constants import MAX_FAILURE_CHARS
from app.ingestion.embed.models import EmbedOutcome
from app.ingestion.enums import DocChange, IngestRunStatus, Stage

UNITS = {
    Stage.DISCOVER: "documents",
    Stage.FETCH: "documents",
    Stage.PARSE: "documents",
    Stage.CHUNK: "chunks",
    Stage.EMBED: "chunks",
}
"""What each stage's counts count, since a run's documents and its chunks read alike."""


class DocumentOutcome(FrozenModel):
    """What one document's pass through the loop committed, or the stage that stopped it."""

    celex: str
    change: DocChange | None = None
    chunks: ChunkCounts = ChunkCounts()
    failed: Stage | None = None
    error: str = ""


class IngestRunResult(BaseModel):
    """Outcome of one ingest run: its discovery diff, its documents, and its embedding pass."""

    run_id: int
    corpus_version: str | None = None
    dropped: list[str] = Field(default_factory=list)
    documents: list[DocumentOutcome] = Field(default_factory=list)
    pruned: int = 0
    embed: EmbedOutcome = Field(default_factory=EmbedOutcome)

    def failures(self, stage: Stage) -> dict[str, str]:
        """Why each document this stage could not process failed."""
        if stage is Stage.EMBED:
            return self.embed.failures
        return {doc.celex: doc.error for doc in self.documents if doc.failed is stage}

    @property
    def committed(self) -> list[DocumentOutcome]:
        """The documents that got all the way through the loop, which are the ones that count."""
        return [doc for doc in self.documents if doc.failed is None]

    @property
    def changes(self) -> Counter[DocChange]:
        return Counter(doc.change for doc in self.committed if doc.change is not None)

    @property
    def chunks(self) -> ChunkCounts:
        """Every document's reconciliation, plus the corpus-wide prune that follows the loop."""
        counts = [doc.chunks for doc in self.committed]
        return ChunkCounts(
            added=sum(count.added for count in counts),
            deleted=sum(count.deleted for count in counts) + self.pruned,
            kept=sum(count.kept for count in counts),
            refreshed=sum(count.refreshed for count in counts),
        )

    @property
    def counts(self) -> dict[Stage, dict[str, int]]:
        """Each stage's outcome buckets, in the order the report and the summary present them."""
        changes = self.changes
        return {
            Stage.DISCOVER: {"dropped": len(self.dropped)},
            Stage.FETCH: {change.value: changes[change] for change in DocChange},
            Stage.PARSE: {"parsed": len(self.committed)},
            Stage.CHUNK: self.chunks.model_dump(),
            Stage.EMBED: {
                "embedded": self.embed.embedded,
                "already_embedded": self.embed.already_embedded,
            },
        }

    @property
    def corpus_complete(self) -> bool:
        """Every discovered document reached storage, so a celex it lacks is repealed, not lost."""
        return not (self.failures(Stage.FETCH) or self.failures(Stage.PARSE))

    @property
    def ok(self) -> bool:
        return not any(self.failures(stage) for stage in Stage)

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.SUCCESS if self.ok else IngestRunStatus.FAILED

    def report(self) -> dict[str, Any]:
        """The run as its row stores it: each stage's counts, plus why each document failed."""
        return {
            stage.value: {
                **counts,
                "failed": {
                    celex: error[:MAX_FAILURE_CHARS]
                    for celex, error in self.failures(stage).items()
                },
            }
            for stage, counts in self.counts.items()
        }

    def line(self, stage: Stage, *, total: int | None = None) -> str:
        """One stage's counts on a line, carrying the unit those counts are in."""
        counts, unit = self.counts[stage], UNITS[stage]
        if total is None:
            total = len(self.documents) if unit == "documents" else sum(counts.values())
        buckets = ", ".join(f"{value} {label.replace('_', ' ')}" for label, value in counts.items())
        return f"[{stage}] {total} {unit}: {buckets}, {len(self.failures(stage))} failed"

    def details(self) -> list[str]:
        """The per-document lines the stage lines are too short to carry."""
        lines = []
        if self.dropped:
            lines.append(f"discover dropped: {', '.join(sorted(self.dropped))}")
        for change in (DocChange.NEW, DocChange.UPDATED):
            if celexes := sorted(doc.celex for doc in self.committed if doc.change is change):
                lines.append(f"fetch {change}: {', '.join(celexes)}")
        for stage in Stage:
            lines += [
                f"{stage} failed: {celex} ({error})"
                for celex, error in sorted(self.failures(stage).items())
            ]
        return lines

    def summary(self) -> str:
        """The run as the CLI prints it: a line per stage, then the per-document detail."""
        return "\n".join(
            [
                f"run {self.run_id} ({self.corpus_version or 'not stamped'})",
                *(f"  {self.line(stage)}" for stage in Stage),
                *(f"  {line}" for line in self.details()),
            ]
        )


class IngestRunUpdate(BaseModel):
    """Partial update body for an ingest run."""

    status: IngestRunStatus | None = None
    corpus_version: str | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
