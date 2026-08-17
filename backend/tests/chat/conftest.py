"""Chat test fakes shared across the chat test modules."""

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from app.core.config import config


class RecordingChatModel(GenericFakeChatModel):
    """Streams a canned answer with real message chunks, recording each prompt once:
    the fake's _stream is built on its _generate, so that is the one place to record."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def fake_chat_model(answer: str = "Ships must comply [1].") -> RecordingChatModel:
    """A chat model that streams one canned answer."""
    return RecordingChatModel(messages=iter([AIMessage(content=answer)]))


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
