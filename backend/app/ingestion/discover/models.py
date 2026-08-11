"""What discovery found: one candidate act, the document it selects, and the run's diff."""

from typing import ClassVar

from pydantic import Field

from app.core.models import FrozenModel
from app.ingestion.models import StageRunResult


class CandidateAct(FrozenModel):
    """One act the topic query returned, with every binding for it folded together."""

    celex: str
    in_force: str | None = None
    consolidations: frozenset[str] = frozenset()


class DiscoveredDocument(FrozenModel):
    """One document discovery found: what to fetch, and which version to try first."""

    topic: str
    source: str
    celex: str
    candidate_celex: str | None


class DiscoverRunResult(StageRunResult):
    """What discovery found this run, and what the previous run held that it no longer returns."""

    UNCOUNTED: ClassVar[frozenset[str]] = StageRunResult.UNCOUNTED | {"celexes"}

    celexes: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)

    def details(self) -> list[str]:
        dropped = [f"dropped: {', '.join(sorted(self.dropped))}"] if self.dropped else []
        return dropped + super().details()
