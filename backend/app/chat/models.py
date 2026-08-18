"""Chat request, graph state and SSE event values."""

from typing import Annotated, Any, Literal

from langchain_core.messages.ai import UsageMetadata
from pydantic import ConfigDict, Field

from app.chat.enums import ChatEventName
from app.core.models import AppModel, ErrorResponse, FrozenModel
from app.retrieval.models import RetrievedChunk


class ChatRequest(AppModel):
    """The question a caller asks."""

    question: str = Field(min_length=1, max_length=2000)


class ChatState(AppModel):
    """What flows through the graph for one question."""

    question: str
    sources: tuple[RetrievedChunk, ...] = ()
    answer: str = ""
    usage: UsageMetadata | None = None


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
    """One fragment of the answer as the model streams it; the token event's payload."""

    text: str


class ChatEventBase(FrozenModel):
    """One frame of the stream: a fixed event name to narrow on, and that event's data.
    The name is defaulted per event but always sent, so the schema marks it required."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class SourcesEvent(ChatEventBase):
    """Sent once, first: the [n] markers the answer will cite, bound to their chunks."""

    event: Literal[ChatEventName.SOURCES] = ChatEventName.SOURCES
    data: tuple[ChatSource, ...]

    @classmethod
    def from_results(cls, results: tuple[RetrievedChunk, ...]) -> "SourcesEvent":
        """Markers run 1..n in context order, matching the prompt's numbering."""
        return cls(
            data=tuple(
                ChatSource.from_result(marker, result)
                for marker, result in enumerate(results, start=1)
            )
        )


class TokenEvent(ChatEventBase):
    """One fragment of the streamed answer."""

    event: Literal[ChatEventName.TOKEN] = ChatEventName.TOKEN
    data: ChatToken


class DoneEvent(ChatEventBase):
    """The last event of a completed stream."""

    event: Literal[ChatEventName.DONE] = ChatEventName.DONE
    data: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(ChatEventBase):
    """The last event of a failed stream, in the app's one error shape."""

    event: Literal[ChatEventName.ERROR] = ChatEventName.ERROR
    data: ErrorResponse


ChatEvent = Annotated[
    SourcesEvent | TokenEvent | DoneEvent | ErrorEvent, Field(discriminator="event")
]
"""Every frame a chat stream carries, told apart by its event name."""
