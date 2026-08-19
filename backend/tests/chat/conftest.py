"""Chat test fakes shared across the chat test modules."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from app.chat.models import ChatState
from app.core.config import config
from tests.conftest import search_result

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
def two_results(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_search(session, request):
        return (search_result(), search_result(id=2, citation="Article 5(1)"))

    monkeypatch.setattr("app.chat.graph.search", fake_search)


@pytest.fixture(autouse=True)
def no_section_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expansion is a database walk covered in tests/retrieval; here it is switched off,
    so the graph works from exactly what the faked search found."""
    monkeypatch.setattr(config, "EXPAND_SECTIONS", False)


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
