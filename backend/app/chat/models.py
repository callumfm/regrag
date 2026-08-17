"""Chat request and SSE payload values."""

from pydantic import Field

from app.core.models import AppModel, ErrorResponse, FrozenModel
from app.retrieval.models import RetrievedChunk


class ChatRequest(AppModel):
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
    def from_result(cls, marker: int, result: RetrievedChunk) -> "ChatSource":
        """The event payload for one retrieved chunk at one marker position."""
        return cls(
            marker=marker,
            chunk_id=result.id,
            celex=result.celex,
            citation=result.citation,
            title=result.title,
        )


class ChatToken(FrozenModel):
    """The token event's payload: one fragment of the streamed answer."""

    text: str


ChatEvent = tuple[ChatSource, ...] | ChatToken | ErrorResponse
"""What an SSE frame's data holds, by event name: sources, token, error. done carries {}."""
