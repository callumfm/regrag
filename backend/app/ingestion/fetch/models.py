"""Fetch-stage values: what discovery found, which version it resolved to, what the run did."""

from typing import ClassVar

from pydantic import Field

from app.core.models import FrozenModel
from app.ingestion.enums import DocChange
from app.ingestion.models import StageRunResult


class DiscoveredDocument(FrozenModel):
    """One document discovery found: what to fetch, and which version to try first."""

    topic: str
    source: str
    celex: str
    candidate_celex: str | None


class ResolvedVersion(FrozenModel):
    """A version-pinned celex and the HTML URL that served it."""

    resolved_celex: str
    url: str


class FetchRunResult(StageRunResult):
    """The corpus diff one fetch produced against the previous run."""

    UNCOUNTED: ClassVar[frozenset[str]] = StageRunResult.UNCOUNTED | {"discovered"}

    discovered: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)

    def details(self) -> list[str]:
        listed = [
            f"{label}: {', '.join(sorted(celexes))}"
            for label, celexes in (
                ("new", self.new),
                ("changed", self.changed),
                ("dropped", self.dropped),
            )
            if celexes
        ]
        return listed + super().details()

    def record(self, change: DocChange, celex: str) -> None:
        """Append the celex to the bucket its change names."""
        getattr(self, change).append(celex)
