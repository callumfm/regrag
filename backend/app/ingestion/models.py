"""Ingestion domain values: run update body, chunk delta, and the run report."""

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from app.ingestion.enums import DocAction, IngestRunStatus


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


@dataclass
class RunReport:
    """Outcome of one ingest run, bucketed for the CLI diff."""

    run_id: int
    corpus_version: str | None = None
    discovered: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_unchanged: int = 0
    unparsed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.unparsed

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.COMPLETED if self.ok else IngestRunStatus.FAILED

    def record(self, action: DocAction, ref: str) -> None:
        bucket = {
            DocAction.NEW: self.new,
            DocAction.CHANGED: self.changed,
            DocAction.UNCHANGED: self.unchanged,
        }
        bucket[action].append(ref)

    def summary(self) -> str:
        lines = [
            f"run {self.run_id}: {len(self.new)} new, {len(self.changed)} changed, "
            f"{len(self.unchanged)} unchanged, {len(self.dropped)} dropped, "
            f"{len(self.failed)} failed, {len(self.unparsed)} unparsed",
            f"  chunks: +{self.chunks_added} added, -{self.chunks_removed} removed, "
            f"{self.chunks_unchanged} unchanged",
            f"  corpus version: {self.corpus_version or '(not stamped)'}",
        ]
        for label, refs in (
            ("new", self.new),
            ("changed", self.changed),
            ("dropped", self.dropped),
        ):
            if refs:
                lines.append(f"  {label}: {', '.join(sorted(refs))}")
        for ref, error in sorted(self.failed.items()):
            lines.append(f"  failed: {ref} ({error})")
        for ref, error in sorted(self.unparsed.items()):
            lines.append(f"  unparsed: {ref} ({error})")
        return "\n".join(lines)
