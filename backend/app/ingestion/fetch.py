"""Corpus fetch: version-diff against the previous run, download only what changed."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.ingestion.discover import DocumentSpec


class DocAction(StrEnum):
    """How a discovered document compares to the previous run."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


def classify(prev_resolved_ref: str | None, resolved_ref: str) -> DocAction:
    if prev_resolved_ref is None:
        return DocAction.NEW
    if prev_resolved_ref != resolved_ref:
        return DocAction.CHANGED
    return DocAction.UNCHANGED


def dropped_refs(specs: Sequence[DocumentSpec], baseline_refs: Iterable[str]) -> list[str]:
    """Baseline refs no longer present in discovery (repealed or out of force)."""
    discovered = {spec.ref for spec in specs}
    return sorted(set(baseline_refs) - discovered)


@dataclass
class RunReport:
    """Outcome of one fetch run, bucketed for the CLI diff."""

    run_id: int
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

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
            f"{len(self.failed)} failed"
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
        return "\n".join(lines)
