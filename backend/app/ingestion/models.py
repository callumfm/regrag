"""Ingestion domain values: the stage-delta base, the three stage deltas, and the run report."""

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field

from app.ingestion.enums import DocAction, IngestRunStatus


class IngestStageDelta(BaseModel):
    """One ingest stage's outcome; subclasses declare their buckets and fill counts()."""

    failed: dict[str, str] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    def counts(self) -> dict[str, int]:
        """This stage's outcome buckets, in reporting order."""
        raise NotImplementedError

    def summary(self) -> str:
        """One line of counts, always closing with the failure count."""
        counted = [f"{value} {label}" for label, value in self.counts().items()]
        return ", ".join([*counted, f"{len(self.failed)} failed"])

    def details(self) -> list[str]:
        """The per-ref lines the summary is too short to carry."""
        return [f"failed: {ref} ({error})" for ref, error in sorted(self.failed.items())]

    def __add__(self, other: Self) -> Self:
        """Combine same-type operands field by field; fields must be int, list or dict."""
        if type(other) is not type(self):
            return NotImplemented
        merged: dict[str, Any] = {}
        for name in type(self).model_fields:
            mine, theirs = getattr(self, name), getattr(other, name)
            if isinstance(mine, dict):
                merged[name] = {**mine, **theirs}
            elif isinstance(mine, list | int) and not isinstance(mine, bool):
                merged[name] = mine + theirs
            else:
                raise TypeError(
                    f"{type(self).__name__}.{name}: delta fields must be int, list or dict"
                )
        return type(self)(**merged)


class FetchDelta(IngestStageDelta):
    """The corpus diff one fetch produced against the previous run."""

    discovered: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "new": len(self.new),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "dropped": len(self.dropped),
        }

    def details(self) -> list[str]:
        listed = [
            f"{label}: {', '.join(sorted(refs))}"
            for label, refs in (
                ("new", self.new),
                ("changed", self.changed),
                ("dropped", self.dropped),
            )
            if refs
        ]
        return listed + super().details()

    def record(self, action: DocAction, ref: str) -> None:
        """Route a document's fetch outcome to its bucket."""
        bucket = {
            DocAction.NEW: self.new,
            DocAction.CHANGED: self.changed,
            DocAction.UNCHANGED: self.unchanged,
        }
        bucket[action].append(ref)


class ParseDelta(IngestStageDelta):
    """Which fetched documents yielded a section tree."""

    parsed: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"parsed": len(self.parsed)}


class ChunkDelta(IngestStageDelta):
    """What reconciling chunks changed, per document or summed across a run."""

    added: int = 0
    removed: int = 0
    unchanged: int = 0

    def counts(self) -> dict[str, int]:
        return {"added": self.added, "removed": self.removed, "unchanged": self.unchanged}


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
        return {"fetch": self.fetch, "parse": self.parse, "chunk": self.chunk}

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
