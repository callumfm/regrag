"""Fetch-stage values: which version a download resolved to, and what the run did."""

from dataclasses import dataclass
from datetime import datetime

from app.core.models import FrozenModel
from app.ingestion.fetch.schemas import RawDocument


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
