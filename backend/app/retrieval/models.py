"""Retrieval values: what a caller asks for, and what comes back."""

from pydantic import Field, model_validator

from app.core.models import FrozenModel
from app.ingestion.chunk.models import Reference


class SearchFilters(FrozenModel):
    """Which slice of the corpus a search may draw from."""

    celex: str | None = None
    topic: str | None = None


class SearchRequest(FrozenModel):
    """What to search the corpus for, where, and how many answers to bring back."""

    query: str
    filters: SearchFilters = SearchFilters()
    limit: int | None = Field(default=None, ge=1)
    """None leaves the count to SEARCH_DEFAULT_LIMIT, read when the search runs."""


class ReferenceTarget(FrozenModel):
    """Which division of which act to look up, named as a citation names it."""

    celex: str
    article: str | None = None
    paragraph: str | None = None
    annex: str | None = None

    @classmethod
    def from_reference(cls, reference: Reference, *, citing: str) -> "ReferenceTarget":
        """The division a stored reference names, in the citing act unless it names another."""
        return cls(
            celex=reference.instrument or citing,
            article=reference.article,
            paragraph=reference.paragraph,
            annex=reference.annex,
        )

    @model_validator(mode="after")
    def _addresses_a_division(self) -> "ReferenceTarget":
        """Following a whole act is a filtered search, not a lookup, so an act alone is refused."""
        if self.article is None and self.annex is None:
            raise ValueError(f"following {self.celex} needs an article or annex, not the act alone")
        return self


class RetrievedChunk(FrozenModel):
    """A chunk as a caller sees it, without the vectors it is found by."""

    id: int
    celex: str
    topic: str
    citation: str
    title: str | None
    text: str
    references: tuple[Reference, ...] = ()


class SearchResult(RetrievedChunk):
    """A retrieved chunk and how it ranked, per leg and fused."""

    score: float
    vector_rank: int | None
    text_rank: int | None
