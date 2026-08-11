"""Fetch-stage values: which version a download resolved to, and what the run did."""

from dataclasses import dataclass
from datetime import datetime

from pydantic import Field

from app.core.models import FrozenModel
from app.ingestion.enums import DocChange
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.models import StageRunResult


class ResolvedVersion(FrozenModel):
    """A version-pinned celex and the HTML URL that served it."""

    resolved_celex: str
    url: str


class StoredBytes(FrozenModel):
    """What a run records about a document's stored bytes."""

    sha256: str
    size_bytes: int
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    """A document's recorded row and the bytes the run has in hand, so parse re-reads nothing."""

    document: RawDocument
    content: bytes


class FetchRunResult(StageRunResult):
    """What one fetch downloaded, against the versions the previous run recorded."""

    new: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)

    def details(self) -> list[str]:
        listed = [
            f"{label}: {', '.join(sorted(celexes))}"
            for label, celexes in (("new", self.new), ("changed", self.changed))
            if celexes
        ]
        return listed + super().details()

    def record(self, change: DocChange, celex: str) -> None:
        """Append the celex to the bucket its change names."""
        getattr(self, change).append(celex)
