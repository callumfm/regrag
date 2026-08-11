"""Retrieval values: what a caller asks for, and what comes back."""

from app.core.models import FrozenModel


class SearchFilters(FrozenModel):
    """Which slice of the corpus a search may draw from."""

    celex: str | None = None
    topic: str | None = None


class RetrievedChunk(FrozenModel):
    """A chunk as a caller sees it, without the vectors it is found by."""

    id: int
    celex: str
    topic: str
    citation: str
    title: str | None
    text: str


class SearchResult(RetrievedChunk):
    """A retrieved chunk and how it ranked, per leg and fused."""

    score: float
    vector_rank: int | None
    text_rank: int | None


class FusedRank(FrozenModel):
    """One chunk's standing after fusion, before its text is loaded."""

    chunk_id: int
    score: float
    vector_rank: int | None
    text_rank: int | None
