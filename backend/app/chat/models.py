"""Chat query, graph state and SSE event values."""

import operator
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from langchain_core.messages.ai import UsageMetadata
from pydantic import ConfigDict, Field, computed_field

from app.chat.enums import (
    ChatEventName,
    ChatNode,
    ChatOutcome,
    ChatStepStatus,
    RefusalReason,
    ToolStep,
)
from app.core.config import config
from app.core.exceptions import DomainError
from app.core.models import AppModel, ErrorResponse, FrozenModel
from app.retrieval.models import RetrievedChunk, SearchResult


class ChatQuery(AppModel):
    """The question a caller asks, and the thread it continues — none on a first question,
    when the server mints one and returns it on the done frame."""

    question: str = Field(min_length=1, max_length=2000)
    thread_id: UUID | None = None


class ChatStepResult(FrozenModel):
    """One step of the path — a graph node, or one tool call a round ran: what it was, how
    long it took, and the tokens it used if it called a model. The shape the ledger persists
    per step, and the trace a run is read back from.

    status: whether the step has finished. Only the stream announces a running one; every step
        the graph appends to the path has returned, so completed is the default.
    ms: how long the step took, which a running one has not spent yet and nothing reads.
    subject: what the step was about where the step alone does not say — the query a search
        ran, the division a follow fetched. Carried to the client, not to the ledger.
    """

    step: ChatNode | ToolStep
    ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    status: ChatStepStatus = ChatStepStatus.COMPLETED
    subject: str | None = None

    @classmethod
    def from_usage(
        cls, step: ChatNode | ToolStep, ms: int, usage: UsageMetadata | None
    ) -> "ChatStepResult":
        """The result of a step that reported usage, or none."""
        if usage is None:
            return cls(step=step, ms=ms)
        return cls(
            step=step,
            ms=ms,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )


class ToolCall(FrozenModel):
    """One tool call assess asked for, as litellm reports it: the tool, and its arguments."""

    name: str
    args: dict[str, Any] = {}


class DecomposedQuestion(FrozenModel):
    """What decompose splits a question into: one search query per thing it asks, in the
    order asked. One query means the question asked one thing."""

    queries: tuple[str, ...]


class ChatTurn(FrozenModel):
    """One earlier turn of the thread as the prompts see it: what was asked, and what was
    answered with its [n] markers stripped, since they numbered that turn's context."""

    question: str
    answer: str


class Refusal(FrozenModel):
    """How a question ended when it ended without an answer: why the refusal was owed, and
    the model's words for it — empty on a gate refusal, which asked no model."""

    reason: RefusalReason
    explanation: str = ""


class StandaloneQuestion(FrozenModel):
    """What rewrite makes of a follow-up: the question restated so that it can be
    searched on its own, naming what the thread's pronouns and shorthand referred to."""

    question: str


class ChatState(AppModel):
    """Everything one question produced: what the graph accumulates as it runs, then what
    only the stream's consumer knows once it ends — how long the request lived, and an error.

    thread_id: the thread the question belongs to, minted here when the caller sent none.
    history: the thread's earlier answered turns, oldest first; empty on a first question.
    standalone_question: the question as rewrite restated it for retrieval, or empty when
        there was nothing to restate or the call failed, so the question as asked is searched.
    steps: the path taken, each node appending its result as it returns and a tool round one
    per call; a sequence, since the loop visits a node more than once.
    queries: the searches decompose split the question into, in the order asked; empty
        when the node was skipped, found one part, or failed, so retrieve searches the
        question as asked.
    hits: what search returned, before the gate and before expansion, kept through a refusal.
    sources: the context blocks that reached the prompt, which the [n] markers number.
    retrieved_sources: how many blocks retrieve left, the base the loop's growth is budgeted
        against; sources grows each round, so the budget cannot be read off it.
    pending_calls: the tool calls assess asked for, not yet executed.
    refusal: why the question ended without an answer, set by the tool round that ran
        assess's refuse call, and by the refuse node itself when nothing was retrieved to
        assess; None on any run that has not refused.
    """

    question: str
    thread_id: UUID = Field(default_factory=uuid4)
    history: tuple[ChatTurn, ...] = ()
    standalone_question: str = ""
    queries: tuple[str, ...] = ()
    steps: Annotated[tuple[ChatStepResult, ...], operator.add] = ()
    hits: tuple[SearchResult, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()
    retrieved_sources: int = 0
    pending_calls: tuple[ToolCall, ...] = ()
    refusal: Refusal | None = None
    answer: str = ""
    total_ms: int | None = None
    error: str | None = None

    @property
    def last_step(self) -> ChatNode | ToolStep | None:
        """The step that just finished — what a values update announces — or None before any."""
        return self.steps[-1].step if self.steps else None

    @property
    def retrieval_question(self) -> str:
        """What retrieval searches: the restated question when rewrite wrote one, else the
        question as asked."""
        return self.standalone_question or self.question

    def token_totals(self) -> tuple[int | None, int | None]:
        """Input and output tokens summed over the steps that reported usage — what the
        request cost — or None for each when none did."""
        inputs = [r.input_tokens for r in self.steps if r.input_tokens is not None]
        outputs = [r.output_tokens for r in self.steps if r.output_tokens is not None]
        return (sum(inputs) if inputs else None, sum(outputs) if outputs else None)

    def sync_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Update this state with the graph's latest snapshot, so one object holds the run
        the ledger records. total_ms and error survive it: no node writes them, so no
        snapshot carries them."""
        fresh = self.model_validate(snapshot)
        for field in snapshot:
            setattr(self, field, getattr(fresh, field))

    def record_error(self, exc: Exception) -> None:
        """The run as failed: named by a DomainError's message, or the type of an unexpected
        one — what the ledger keeps, while the wire says only that something went wrong."""
        self.error = exc.message if isinstance(exc, DomainError) else type(exc).__name__

    def log_fields(self) -> dict[str, Any]:
        """The run as the stats line logs it: everything but the content — which, on a tool
        step, includes what the call was for."""
        content = (
            "question",
            "history",
            "standalone_question",
            "queries",
            "hits",
            "sources",
            "answer",
            "pending_calls",
        )
        exclude_fields: dict[str, Any] = {field: True for field in content} | {
            "steps": {"__all__": {"status", "subject"}}
        }
        fields = self.model_dump(mode="json", exclude=exclude_fields)
        return fields | {
            "hits": len(self.hits),
            "sources": len(self.sources),
            "queries": len(self.queries),
        }

    def assess_rounds(self) -> int:
        """How many times assess has asked — what the loop's budget is spent against. Read
        off assess rather than the tool steps, which number one per call, not per round."""
        return sum(1 for result in self.steps if result.step is ChatNode.ASSESS)

    @property
    def context_settled(self) -> bool:
        """Whether the context is final: retrieval ended with the loop off or the gate
        shut, assess asked for nothing, or the last round consumed the budget or refused."""
        match self.last_step:
            case ChatNode.RETRIEVE:
                return not self.sources or not config.ASSESS_ENABLED
            case ChatNode.ASSESS:
                return not self.pending_calls
            case ToolStep():
                return self.refusal is not None or self.assess_rounds() >= config.ASSESS_MAX_ROUNDS
            case _:
                return False

    @computed_field
    @property
    def outcome(self) -> ChatOutcome:
        """How the run ended, read off the path and the error: a stream that raised, one
        the gate refused, one that answered, or one the client left before either."""
        visited = {result.step for result in self.steps}
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
    text: str

    @classmethod
    def from_result(cls, marker: int, result: RetrievedChunk) -> "ChatSource":
        """The event payload for one retrieved chunk at one marker position."""
        return cls(
            marker=marker,
            chunk_id=result.id,
            celex=result.celex,
            citation=result.citation,
            title=result.title,
            text=result.text,
        )


class ChatThread(FrozenModel):
    """The thread a turn was recorded under, which a follow-up sends back."""

    thread_id: UUID


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


class StepEvent(ChatEventBase):
    """One step of the path, sent as it starts and again as it finishes."""

    event: Literal[ChatEventName.STEP] = ChatEventName.STEP
    data: ChatStepResult


class TextEvent(ChatEventBase):
    """One fragment of the answer's text, as the model streams it — or the whole refusal."""

    event: Literal[ChatEventName.TEXT] = ChatEventName.TEXT
    data: str


class DoneEvent(ChatEventBase):
    """The last event of a completed stream: the thread the turn belongs to."""

    event: Literal[ChatEventName.DONE] = ChatEventName.DONE
    data: ChatThread


class ErrorEvent(ChatEventBase):
    """The last event of a failed stream, in the app's one error shape."""

    event: Literal[ChatEventName.ERROR] = ChatEventName.ERROR
    data: ErrorResponse


ChatEvent = Annotated[
    SourcesEvent | StepEvent | TextEvent | DoneEvent | ErrorEvent, Field(discriminator="event")
]
"""Every frame a chat stream carries, told apart by its event name."""
