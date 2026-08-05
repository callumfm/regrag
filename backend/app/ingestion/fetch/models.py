"""Fetch-stage values: what discovery found and what resolution turned it into."""

from app.core.models import FrozenModel


class DocumentSpec(FrozenModel):
    """One discovered corpus document; transient wire data between discover and fetch."""

    topic: str
    source: str
    ref: str
    candidate_ref: str | None


class Resolution(FrozenModel):
    """A verified resolution: version-pinned ref and its fetchable HTML URL."""

    resolved_ref: str
    url: str
