"""Chat test fakes shared across the chat test modules."""

import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import litellm
import openai
import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from app.chat.graph.service import chat_graph
from app.chat.models import ChatState, ChatTurn
from app.chat.tools.models import ToolCall
from app.core.config import config
from app.retrieval.models import RetrievedChunk, SearchRequest
from tests.conftest import install_chat_model, search_result

USAGE = UsageMetadata(input_tokens=1500, output_tokens=40, total_tokens=1540)


class RecordingChatModel(GenericFakeChatModel):
    """Streams a canned answer with real message chunks, recording each prompt once:
    the fake's _stream is built on its _generate, so that is the one place to record."""

    received: list[list[BaseMessage]] = Field(default_factory=list)
    usage: UsageMetadata | None = None
    """Reported as litellm does: on the message when invoked outright, and as a final
    usage-only chunk after the answer's text when streamed."""

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        result = super()._generate(messages, *args, **kwargs)
        message = result.generations[0].message
        if self.usage and isinstance(message, AIMessage):
            message.usage_metadata = self.usage
        return result

    def _stream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        yield from super()._stream(messages, *args, **kwargs)
        if self.usage:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=self.usage))


def fake_chat_model(answer: str = "Ships must comply [1].") -> RecordingChatModel:
    """A chat model that streams one canned answer and reports USAGE for it."""
    return RecordingChatModel(messages=iter([AIMessage(content=answer)]), usage=USAGE)


def tool_call_message(name: str, args: dict) -> AIMessage:
    """An assess turn asking for one tool, shaped as litellm parses provider tool calls."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}]
    )


THINKING = "weighing the context"
ANSWER = "Ships must comply [1]."


class ReasoningChatModel(GenericFakeChatModel):
    """Streams block-list content, as litellm returns it once the model reasons."""

    def _stream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        for block in (
            {"type": "thinking", "thinking": THINKING},
            {"type": "text", "text": ANSWER},
        ):
            yield ChatGenerationChunk(message=AIMessageChunk(content=[block]))


def reasoning_chat_model() -> ReasoningChatModel:
    """A chat model whose chunks carry content blocks rather than strings."""
    return ReasoningChatModel(messages=iter([AIMessage(content="unused")]))


@pytest.fixture
def one_result(monkeypatch: pytest.MonkeyPatch) -> list[SearchRequest]:
    """Search finds one chunk; the returned list collects what it was asked for."""
    calls: list[SearchRequest] = []

    async def fake_search(session, request):
        calls.append(request)
        return (search_result(),)

    monkeypatch.setattr("app.chat.graph.retrieve.search", fake_search)
    return calls


@pytest.fixture
def two_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(session, request):
        return (search_result(), search_result(id=2, citation="Article 5(1)"))

    monkeypatch.setattr("app.chat.graph.retrieve.search", fake_search)


@pytest.fixture(autouse=True)
def no_section_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expansion is a database walk covered in tests/retrieval; here it is switched off,
    so the graph works from exactly what the faked search found."""
    monkeypatch.setattr(config, "EXPAND_SECTIONS", False)


@pytest.fixture(autouse=True)
def no_assess_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop is off by default so every pre-loop test keeps meaning exactly what it
    said; a loop test takes `loop_on` and fakes assess_model itself."""
    monkeypatch.setattr(config, "ASSESS_ENABLED", False)


@pytest.fixture
def loop_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop back on, with its default two rounds, undoing the autouse switch-off."""
    monkeypatch.setattr(config, "ASSESS_ENABLED", True)
    monkeypatch.setattr(config, "ASSESS_MAX_ROUNDS", 2)


@pytest.fixture(autouse=True)
def no_decompose(monkeypatch: pytest.MonkeyPatch) -> None:
    """The split is off by default so every test that runs the graph keeps meaning exactly
    what it said; a decompose test takes `decompose_on` and fakes decompose_model itself."""
    monkeypatch.setattr(config, "DECOMPOSE_ENABLED", False)


@pytest.fixture
def decompose_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The split back on, undoing the autouse switch-off."""
    monkeypatch.setattr(config, "DECOMPOSE_ENABLED", True)


@pytest.fixture
def assess_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., RecordingChatModel]:
    """Install an assess model answering with the given turns in order, and hand back the
    fake, whose `received` holds the prompts it saw."""

    def install(*turns: AIMessage) -> RecordingChatModel:
        model = RecordingChatModel(messages=iter(turns), usage=USAGE)
        monkeypatch.setattr("app.chat.graph.assess.assess_model", lambda: model)
        return model

    return install


@pytest.fixture
def decompose_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., RecordingChatModel]:
    """Install a decompose model answering with the given turns in order, and hand back
    the fake, whose `received` holds the prompts it saw."""

    def install(*turns: AIMessage) -> RecordingChatModel:
        model = RecordingChatModel(messages=iter(turns), usage=USAGE)
        monkeypatch.setattr("app.chat.graph.decompose.decompose_model", lambda: model)
        return model

    return install


def split_message(*queries: str) -> AIMessage:
    """A decompose turn, shaped as a model bound to the DecomposedQuestion format answers."""
    return AIMessage(content=json.dumps({"queries": list(queries)}))


@pytest.fixture
def rewrite_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., RecordingChatModel]:
    """Install a rewrite model answering with the given turns in order, and hand back
    the fake, whose `received` holds the prompts it saw."""

    def install(*turns: AIMessage) -> RecordingChatModel:
        model = RecordingChatModel(messages=iter(turns), usage=USAGE)
        monkeypatch.setattr("app.chat.graph.rewrite.rewrite_model", lambda: model)
        return model

    return install


def restated_message(question: str) -> AIMessage:
    """A rewrite turn, shaped as a model bound to the StandaloneQuestion format answers."""
    return AIMessage(content=json.dumps({"question": question}))


@pytest.fixture
def tool_results(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[ToolCall]]:
    """Install a run_tool_call answering every call with the given chunks, and hand back
    the list the calls it received accumulate in."""

    def install(*found: RetrievedChunk) -> list[ToolCall]:
        calls: list[ToolCall] = []

        async def fake_run_tool_call(call):
            calls.append(call)
            return found

        monkeypatch.setattr("app.chat.graph.assess.run_tool_call", fake_run_tool_call)
        return calls

    return install


@pytest.fixture(autouse=True)
def no_tool_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool call opens its own session; the faked tool paths never touch it, so a null
    one stands in and no chat test reaches the database."""

    @asynccontextmanager
    async def no_session(**kwargs: Any) -> AsyncIterator[None]:
        yield None

    monkeypatch.setattr("app.chat.tools.service.get_session", no_session)


@pytest.fixture(autouse=True)
def recorded_requests(monkeypatch: pytest.MonkeyPatch) -> list[ChatState]:
    """Capture the state stream_chat_events hands to create_chat_request, and give it no session
    to hand over: the write is covered in test_service, so no streaming test needs the
    database."""
    states: list[ChatState] = []

    @asynccontextmanager
    async def no_session(**kwargs: Any) -> AsyncIterator[None]:
        yield None

    async def fake_create_chat_request(session: None, state: ChatState) -> None:
        states.append(state)

    monkeypatch.setattr("app.chat.stream.get_session", no_session)
    monkeypatch.setattr("app.chat.stream.create_chat_request", fake_create_chat_request)
    return states


QUESTION = "What is the GHG intensity limit?"


def rate_limited() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.example")
    return openai.RateLimitError(
        message="provider said no", response=httpx.Response(429, request=request), body=None
    )


class FailingModel(RecordingChatModel):
    """Refuses the first `failures` prompts as a rate limit, then answers."""

    failures: int = 1

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        if len(self.received) < self.failures:
            self.received.append(list(messages))
            raise rate_limited()
        return super()._generate(messages, *args, **kwargs)


def streamed_text(data: Any) -> str:
    """The text of one messages-mode stream item, a (chunk, metadata) pair."""
    chunk, _ = data
    return chunk.text


def litellm_stream(
    monkeypatch, *deltas: dict[str, Any], usage: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Stand litellm's completion call in with these deltas — and, as litellm reports it
    when asked, a trailing usage-only chunk; the calls made are returned."""
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        calls.append(kwargs)

        async def chunks() -> AsyncIterator[dict[str, Any]]:
            for delta in deltas:
                yield {"choices": [{"delta": delta, "finish_reason": None}]}
            if usage:
                yield {"choices": [], "usage": usage}

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


def litellm_completion(monkeypatch, content: str, usage: dict[str, int]) -> list[dict[str, Any]]:
    """Stand litellm's completion call in for one blocking answer, returning the calls made."""
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": usage,
        }

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


@pytest.fixture
def answer_model(monkeypatch):
    model = fake_chat_model("Answered [1].")
    install_chat_model(monkeypatch, lambda *_: model)
    return model


async def run_graph() -> ChatState:
    """The graph run, folded back onto the state it started from."""
    state = ChatState(question=QUESTION)
    state.sync_from_snapshot(await chat_graph.ainvoke(state))
    return state


def hits_for(**per_query: tuple) -> tuple[Callable, list[SearchRequest]]:
    """A search answering each query with its own hits, recording the requests made."""
    requests: list[SearchRequest] = []

    async def fake_search(session, request):
        requests.append(request)
        return per_query[request.query]

    return fake_search, requests


FOLLOW_UP = "What penalties does it impose?"
RESTATED = "What penalties does FuelEU Maritime impose?"
HISTORY = (ChatTurn(question="What is FuelEU Maritime?", answer="A regulation on fuel."),)
