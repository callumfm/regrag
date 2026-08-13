"""Fetch-stage values: which version a download resolved to, and what the run did."""

from dataclasses import dataclass
from datetime import datetime

from app.core.models import FrozenModel
from app.ingestion.fetch.schemas import RawDocument


class ResolvedVersion(FrozenModel):
    """A version-pinned celex and the HTML URL that served it."""

    resolved_celex: str
    url: str


class FetchedVersion(ResolvedVersion):
    """A resolved version, plus what the run records about the bytes it holds for it.

    sha256: their fingerprint, which also keys them in the store.
    size_bytes: their length.
    fetched_at: when they were downloaded, carried over unchanged when bytes are reused.
    """

    sha256: str
    size_bytes: int
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    """A document's recorded row and the bytes the run has in hand, so parse re-reads nothing."""

    document: RawDocument
    content: bytes


class RawDocsQuery(FrozenModel):
    """Filters for the standing corpus; None leaves a filter off, [] genuinely matches nothing."""

    include_topics: list[str] | None = None
    exclude_topics: list[str] | None = None
