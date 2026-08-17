"""Chat graph: state flow, search passthrough, prompt assembly, error wrapping."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import litellm
import openai
import pytest
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult

from app.chat.graph import ChatState, chat_graph
from app.core.config import config
from app.core.llm import LLMError
from app.retrieval.models import SearchRequest
from tests.chat.conftest import ANSWER, THINKING, RecordingChatModel, fake_chat_model
from tests.conftest import search_result

pytestmark = pytest.mark.anyio

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


@pytest.fixture
def one_result(monkeypatch):
    calls: list[SearchRequest] = []

    async def fake_search(session, request):
        calls.append(request)
        return (search_result(),)

    monkeypatch.setattr("app.chat.graph.search", fake_search)
    return calls


async def test_graph_retrieves_then_answers(one_result, monkeypatch):
    model = fake_chat_model("Yes, Article 4 [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == "Yes, Article 4 [1]."
    assert state["sources"] == (search_result(),)
    assert one_result == [SearchRequest(query=QUESTION, limit=config.CHAT_SOURCES)]


async def test_model_receives_system_prompt_and_numbered_context(monkeypatch):
    async def fake_search(session, request):
        return (search_result(text="A very specific clause."),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", fake_search)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    await chat_graph.ainvoke(ChatState(question=QUESTION))

    (prompt,) = model.received
    assert isinstance(prompt[0], SystemMessage)
    assert "[1] (32023R1805, Article 4(1))" in prompt[1].content
    assert "A very specific clause." in prompt[1].content


async def test_a_transient_provider_failure_is_retried(one_result, monkeypatch):
    model = FailingModel(messages=iter(["Second time lucky [1]."]), failures=1)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == "Second time lucky [1]."
    assert len(model.received) == 2


async def test_a_persistent_provider_failure_becomes_a_transient_llm_error(one_result, monkeypatch):
    model = FailingModel(messages=iter([]), failures=10)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    with pytest.raises(LLMError) as exc_info:
        await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert exc_info.value.transient is True
    assert str(exc_info.value) == "chat call failed"
    assert len(model.received) == 3


def streamed_text(data: Any) -> str:
    """The text of one messages-mode stream item, a (chunk, metadata) pair."""
    chunk, _ = data
    return chunk.text


def litellm_stream(monkeypatch, *deltas: dict[str, Any]) -> list[dict[str, Any]]:
    """Stand litellm's completion call in with these deltas; the calls made are returned."""
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        calls.append(kwargs)

        async def chunks() -> AsyncIterator[dict[str, Any]]:
            for delta in deltas:
                yield {"choices": [{"delta": delta, "finish_reason": None}]}

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


async def test_the_chat_client_streams_one_token_per_litellm_delta(one_result, monkeypatch):
    """The seam below the fakes: chat_model()'s ChatLiteLLM asks litellm to stream, and
    the graph's message stream sees each delta as it lands."""
    calls = litellm_stream(
        monkeypatch, {"role": "assistant", "content": "Ships must "}, {"content": "comply [1]."}
    )

    texts = [
        streamed_text(data)
        async for mode, data in chat_graph.astream(
            ChatState(question=QUESTION), stream_mode=["updates", "messages"]
        )
        if mode == "messages"
    ]

    assert calls[0]["stream"] is True
    assert calls[0]["model"] == config.CHAT_MODEL
    assert [text for text in texts if text] == ["Ships must ", "comply [1]."]


async def test_the_chat_client_answers_with_the_text_of_a_reasoning_response(
    one_result, monkeypatch
):
    """litellm's reasoning_content becomes a thinking block ahead of the text; the answer
    is the text alone, not the repr of the block list."""
    litellm_stream(
        monkeypatch,
        {"role": "assistant", "content": "", "reasoning_content": THINKING},
        {"content": ANSWER},
    )

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["answer"] == ANSWER


async def test_retrieve_widens_what_search_found_to_whole_sections(one_result, monkeypatch):
    """The graph hands search's hits to expansion, so the prompt sees whole sections."""
    widened = (search_result(id=2, citation="Article 4(2)", text="The limit is 91,16 gCO2e/MJ."),)

    async def fake_expand(session, chunks, *, limit):
        assert tuple(chunks) == (search_result(),)
        assert limit == config.CHAT_CONTEXT_CHUNKS
        return widened

    monkeypatch.setattr(config, "EXPAND_SECTIONS", True)
    monkeypatch.setattr("app.chat.graph.expand_sections", fake_expand)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["sources"] == widened


async def test_retrieve_leaves_search_alone_when_expansion_is_off(one_result, monkeypatch):
    """The off switch skips the widening query, not just its result."""

    async def refuse(session, chunks, *, limit):
        raise AssertionError("expansion ran with EXPAND_SECTIONS off")

    monkeypatch.setattr("app.chat.graph.expand_sections", refuse)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: fake_chat_model())

    state = await chat_graph.ainvoke(ChatState(question=QUESTION))

    assert state["sources"] == (search_result(),)
