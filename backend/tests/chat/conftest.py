"""Chat test fakes shared across the chat test modules."""

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from app.core.config import config

USAGE = UsageMetadata(input_tokens=1500, output_tokens=40, total_tokens=1540)


class RecordingChatModel(GenericFakeChatModel):
    """Streams a canned answer with real message chunks, recording each prompt once:
    the fake's _stream is built on its _generate, so that is the one place to record.
    Usage is reported the way litellm reports it: on the message when invoked, as a
    usage-only final chunk when streamed."""

    received: list[list[BaseMessage]] = Field(default_factory=list)
    usage: UsageMetadata | None = None

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        result = super()._generate(messages, *args, **kwargs)
        message = result.generations[0].message
        if isinstance(message, AIMessage):
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


@pytest.fixture(autouse=True)
def no_section_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expansion is a database walk covered in tests/retrieval; here it is switched off,
    so the graph works from exactly what the faked search found."""
    monkeypatch.setattr(config, "EXPAND_SECTIONS", False)
