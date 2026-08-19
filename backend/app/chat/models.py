"""Chat query, graph state and SSE event values."""

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages.ai import UsageMetadata
from pydantic import ConfigDict, Field, computed_field

from app.chat.enums import ChatEventName, ChatNode, ChatOutcome
from app.core.exceptions import DomainError
from app.core.models import AppModel, ErrorResponse, FrozenModel
from app.retrieval.models import RetrievedChunk, SearchResult


class ChatQuery(AppModel):
    """The question a caller asks."""

    question: str = Field(min_length=1, max_length=2000)


class ChatNodeResult(FrozenModel):
    """One step of the path: the node, how long it took, and the tokens it used if it
    called a model — the shape the ledger persists per node."""

    node: ChatNode
    ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None

    @classmethod
    def from_usage(cls, node: ChatNode, ms: int, usage: UsageMetadata | None) -> "ChatNodeResult":
        """The result of a node that reported usage, or none."""
        if usage is None:
            return cls(node=node, ms=ms)
        return cls(
            node=node,
            ms=ms,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )


class ChatState(AppModel):
    """Everything one question produced: what the graph accumulates as it runs, then what
    only the stream's consumer knows once it ends — how long the request lived, and an error.

    nodes: the path taken, each node appending its result as it returns; a sequence, since a
    tool loop will visit a node more than once.
    hits: what search returned, before the gate and before expansion, kept through a refusal.
    sources: the context blocks that reached the prompt, which the [n] markers number.
    """

    question: str
    nodes: Annotated[tuple[ChatNodeResult, ...], operator.add] = ()
    hits: tuple[SearchResult, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()
    answer: str = ""
    total_ms: int | None = None
    error: str | None = None

    @property
    def last_node(self) -> ChatNode | None:
        """The node that just returned — what a values step announces — or None before any."""
        return self.nodes[-1].node if self.nodes else None

    def token_totals(self) -> tuple[int | None, int | None]:
        """Input and output tokens summed over the nodes that reported usage — what the
        request cost — or None for each when none did."""
        inputs = [r.input_tokens for r in self.nodes if r.input_tokens is not None]
        outputs = [r.output_tokens for r in self.nodes if r.output_tokens is not None]
        return (sum(inputs) if inputs else None, sum(outputs) if outputs else None)

    def refresh(self, snapshot: dict[str, Any]) -> None:
        """The graph's state as it now stands, folded onto this object: the values stream
        hands back a fresh dict each step, and the record wants one object per run. Only
        the fields the snapshot carries are touched; what the consumer sets stays."""
        fresh = self.model_validate(snapshot)
        for field in snapshot:
            setattr(self, field, getattr(fresh, field))

    def record_error(self, exc: Exception) -> None:
        """The run as failed: named by a DomainError's message, or the type of an unexpected
        one — what the ledger keeps, while the wire says only that something went wrong."""
        self.error = exc.message if isinstance(exc, DomainError) else type(exc).__name__

    def log_fields(self) -> dict[str, Any]:
        """The run as the stats line logs it: everything but the content."""
        fields = self.model_dump(mode="json", exclude={"question", "hits", "sources", "answer"})
        return fields | {"hits": len(self.hits), "sources": len(self.sources)}

    @computed_field
    @property
    def outcome(self) -> ChatOutcome:
        """How the run ended, read off the path and the error: a stream that raised, one
        the gate refused, one that answered, or one the client left before either."""
        visited = {result.node for result in self.nodes}
        if self.error:
            return ChatOutcome.ERROR
        if ChatNode.REFUSE in visited:
            return ChatOutcome.REFUSED
        if ChatNode.SYNTHESIZE in visited:
            return ChatOutcome.DONE
        return ChatOutcome.ABORTED


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


class ChatEventBase(FrozenModel):
    """One frame of the stream: which event, and that event's data. Each event narrows
    `event` to its own name — what the union discriminates on — defaulted but always sent,
    so the schema marks it required."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    event: ChatEventName


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


class TextEvent(ChatEventBase):
    """One fragment of the answer's text, as the model streams it — or the whole refusal."""

    event: Literal[ChatEventName.TEXT] = ChatEventName.TEXT
    data: str


class DoneEvent(ChatEventBase):
    """The last event of a completed stream."""

    event: Literal[ChatEventName.DONE] = ChatEventName.DONE
    data: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(ChatEventBase):
    """The last event of a failed stream, in the app's one error shape."""

    event: Literal[ChatEventName.ERROR] = ChatEventName.ERROR
    data: ErrorResponse


ChatEvent = Annotated[
    SourcesEvent | TextEvent | DoneEvent | ErrorEvent, Field(discriminator="event")
]
"""Every frame a chat stream carries, told apart by its event name."""
