"""Fetch-stage values: which version a download resolved to, and what the run did."""

from dataclasses import dataclass
from datetime import datetime

from app.core.models import FrozenModel
from app.ingestion.fetch.schemas import RawDocument

HTML_URL_TEMPLATE = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"


def eurlex_html_url(version_celex: str) -> str:
    """The EUR-Lex HTML endpoint for one version."""
    return HTML_URL_TEMPLATE.format(celex=version_celex)


class FetchedVersion(FrozenModel):
    """The version a download resolved to, and what the run records about the bytes it holds.

    resolved_celex: the version EUR-Lex served, which is a candidate or the original act.
    sha256: their fingerprint, which also keys them in the store.
    size_bytes: their length.
    fetched_at: when they were downloaded, carried over unchanged when bytes are reused.
    """

    resolved_celex: str
    sha256: str
    size_bytes: int
    fetched_at: datetime

    @property
    def url(self) -> str:
        """Where EUR-Lex served this version; derived, so no row can disagree with its own celex."""
        return eurlex_html_url(self.resolved_celex)


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    """A document's recorded row and the bytes the run has in hand, so parse re-reads nothing."""

    document: RawDocument
    content: bytes


class RawDocsQuery(FrozenModel):
    """Filters for the standing corpus; None leaves a filter off, [] genuinely matches nothing."""

    include_topics: list[str] | None = None
    exclude_topics: list[str] | None = None
