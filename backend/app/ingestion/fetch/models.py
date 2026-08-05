"""Fetch-stage values: what discovery found and what resolution turned it into."""

from pydantic import Field

from app.core.models import FrozenModel
from app.ingestion.enums import DocAction
from app.ingestion.stage import IngestStageDelta


class DiscoveredDocument(FrozenModel):
    """One document discovery found: what to fetch, and which version to try first."""

    topic: str
    source: str
    ref: str
    candidate_ref: str | None


class Resolution(FrozenModel):
    """A verified resolution: version-pinned ref and its fetchable HTML URL."""

    resolved_ref: str
    url: str


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
