"""Chat graph: state flow, search passthrough, prompt assembly, error wrapping."""

import os
from typing import Any

import httpx
import openai
import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_litellm import ChatLiteLLM

from app.chat.graph import chat_graph
from app.core.config import config
from app.core.llm import LLMError
from tests.chat.conftest import RecordingChatModel, fake_chat_model, make_result

pytestmark = pytest.mark.anyio

RUN_CONFIG: RunnableConfig = {"configurable": {"session": None}}


def graph_input(question: str = "What is the GHG intensity limit?") -> dict:
    return {"question": question, "sources": (), "answer": ""}


async def test_graph_retrieves_then_answers(monkeypatch):
    calls: list[tuple[str, int]] = []

    async def fake_search(session, query, **kwargs):
        calls.append((query, kwargs["limit"]))
        return (make_result(),)

    monkeypatch.setattr("app.chat.graph.search", fake_search)
    monkeypatch.setattr("app.chat.graph.chat_model", fake_chat_model("Yes, Article 4 [1]."))

    state = await chat_graph.ainvoke(graph_input(), config=RUN_CONFIG)

    assert state["answer"] == "Yes, Article 4 [1]."
    assert state["sources"] == (make_result(),)
    assert calls == [("What is the GHG intensity limit?", config.CHAT_CONTEXT_CHUNKS)]


async def test_model_receives_system_prompt_and_numbered_context(monkeypatch):
    async def fake_search(session, query, **kwargs):
        return (make_result(text="A very specific clause."),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", fake_search)
    monkeypatch.setattr("app.chat.graph.chat_model", model)

    await chat_graph.ainvoke(graph_input(), config=RUN_CONFIG)

    (prompt,) = model.received
    assert isinstance(prompt[0], SystemMessage)
    assert "[1] (32023R1805, Article 4(1))" in prompt[1].content
    assert "A very specific clause." in prompt[1].content


async def test_provider_failure_becomes_transient_llm_error(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.example")

    class FailingModel(RecordingChatModel):
        def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
            raise openai.RateLimitError(
                message="provider said no",
                response=httpx.Response(429, request=request),
                body=None,
            )

        def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
            raise openai.RateLimitError(
                message="provider said no",
                response=httpx.Response(429, request=request),
                body=None,
            )

    async def fake_search(session, query, **kwargs):
        return (make_result(),)

    monkeypatch.setattr("app.chat.graph.search", fake_search)
    monkeypatch.setattr("app.chat.graph.chat_model", FailingModel(messages=iter([])))

    with pytest.raises(LLMError) as exc_info:
        await chat_graph.ainvoke(graph_input(), config=RUN_CONFIG)

    assert exc_info.value.transient is True


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs a real Anthropic key")
async def test_real_chat_model_streams_string_chunks():
    model = ChatLiteLLM(
        model=config.CHAT_MODEL,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=32,
        request_timeout=30,
    )
    texts = [
        chunk.content
        async for chunk in model.astream([HumanMessage("Reply with the single word OK.")])
    ]
    assert all(isinstance(text, str) for text in texts)
    assert "".join(texts)  # ty: ignore[no-matching-overload]
