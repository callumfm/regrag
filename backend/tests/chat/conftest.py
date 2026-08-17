"""Chat test fakes shared across the chat test modules."""

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from app.retrieval.models import SearchResult


def make_result(**overrides: Any) -> SearchResult:
    """A retrieved chunk with sane defaults, overridable per field."""
    defaults: dict[str, Any] = {
        "id": 1,
        "celex": "32023R1805",
        "topic": "fueleu",
        "citation": "Article 4(1)",
        "title": "Greenhouse gas intensity limit",
        "text": "The greenhouse gas intensity of the energy used on board.",
        "score": 0.9,
        "vector_rank": 1,
        "text_rank": 1,
    }
    return SearchResult(**{**defaults, **overrides})


class RecordingChatModel(GenericFakeChatModel):
    """Streams a canned answer with real message chunks, recording each prompt."""

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def _stream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        self.received.append(list(messages))
        yield from super()._stream(messages, *args, **kwargs)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def fake_chat_model(answer: str = "Ships must comply [1].") -> RecordingChatModel:
    """A chat model that streams one canned answer."""
    return RecordingChatModel(messages=iter([AIMessage(content=answer)]))


THINKING = "weighing the context"


class ReasoningChatModel(GenericFakeChatModel):
    """Streams block-list content, as litellm returns it once the model reasons."""

    def _stream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        for block in (
            {"type": "thinking", "thinking": THINKING},
            {"type": "text", "text": "Ships must comply [1]."},
        ):
            yield ChatGenerationChunk(message=AIMessageChunk(content=[block]))


def reasoning_chat_model() -> ReasoningChatModel:
    """A chat model whose chunks carry content blocks rather than strings."""
    return ReasoningChatModel(messages=iter([AIMessage(content="unused")]))
