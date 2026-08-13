"""Chat request and SSE payload values."""

from pydantic import BaseModel, Field

from app.core.models import FrozenModel
from app.retrieval.models import SearchResult


class ChatRequest(BaseModel):
    """The question a caller asks."""

    question: str = Field(min_length=1, max_length=2000)


class ChatSource(FrozenModel):
    """One context block as the sources event reports it, binding marker to chunk."""

    marker: int
    chunk_id: int
    celex: str
    citation: str
    title: str | None

    @classmethod
    def from_result(cls, marker: int, result: SearchResult) -> "ChatSource":
        """The event payload for one retrieved chunk at one marker position."""
        return cls(
            marker=marker,
            chunk_id=result.id,
            celex=result.celex,
            citation=result.citation,
            title=result.title,
        )
