"""Chat stream orchestration: every way a stream ends records one request with what it reached."""

import json
import logging
import re

import anyio
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from sqlalchemy.exc import OperationalError

from app.chat.enums import ChatNode, ChatOutcome
from app.chat.models import DoneEvent, ErrorEvent, SourcesEvent, TextEvent
from app.chat.prompts import REFUSAL_ANSWER
from app.chat.stream import stream_chat_events
from app.core.llm import LLMError
from tests.chat.conftest import USAGE, RecordingChatModel, fake_chat_model, tool_call_message
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


async def test_finished_stream_records_timings_sources_and_usage(
    two_results, monkeypatch, recorded_requests
):
    model = fake_chat_model("Two words [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda *_: model)

    async for _ in stream_chat_events("q"):
        pass

    [state] = recorded_requests
    assert state.question == "q"
    assert state.outcome is ChatOutcome.DONE
    retrieve, synthesize = state.steps
    assert (retrieve.step, synthesize.step) == (ChatNode.RETRIEVE, ChatNode.SYNTHESIZE)
    assert (retrieve.input_tokens, retrieve.output_tokens) == (None, None)
    assert (synthesize.input_tokens, synthesize.output_tokens) == (1500, 40)
    assert len(state.sources) == 2
    assert state.total_ms is not None
    assert 0 <= sum(result.ms for result in state.steps) <= state.total_ms
    assert state.error is None


async def test_failed_stream_records_what_it_reached(monkeypatch, recorded_requests):
    async def failing_search(session, request):
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.chat.graph.search", failing_search)

    async for _ in stream_chat_events("q"):
        pass

    [state] = recorded_requests
    assert state.outcome is ChatOutcome.ERROR
    assert state.error == "embedding call failed"
    assert state.steps == ()
    assert state.sources == ()
    assert state.token_totals() == (None, None)


async def test_unexpected_failure_is_recorded_by_its_type_and_sent_as_the_generic_error(
    monkeypatch, recorded_requests
):
    """The wire says only that something went wrong; the ledger keeps what did."""

    async def exploding_search(session, request):
        raise RuntimeError("pool exhausted")

    monkeypatch.setattr("app.chat.graph.search", exploding_search)

    events = [event async for event in stream_chat_events("q")]

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].data.message == "An unexpected error occurred"
    [state] = recorded_requests
    assert state.outcome is ChatOutcome.ERROR
    assert state.error == "RuntimeError"


async def test_abandoned_stream_still_records(two_results, monkeypatch, recorded_requests):
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda *_: model)

    events = stream_chat_events("q")
    await anext(events)
    await events.aclose()

    [state] = recorded_requests
    assert state.outcome is ChatOutcome.ABORTED
    assert len(state.sources) == 2
    assert [result.step for result in state.steps] == [ChatNode.RETRIEVE]


async def test_failed_write_is_logged_not_raised(two_results, monkeypatch, caplog):
    """The ledger write failing after the answer went out is a log line, not a broken stream."""
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda *_: model)

    async def broken_create_chat_request(session, state):
        raise OperationalError("insert", {}, ConnectionRefusedError("database away"))

    monkeypatch.setattr("app.chat.stream.create_chat_request", broken_create_chat_request)

    events = [event async for event in stream_chat_events("q")]

    assert events[-1] == DoneEvent()
    [error] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error.getMessage() == "chat request not recorded"
    assert error.exc_info is not None


async def test_cancelled_stream_still_records(two_results, monkeypatch, recorded_requests):
    """A client leaving cancels the streaming task mid-stream; the record still lands.
    The fake write yields once, so an unshielded await there would be cancelled, not run."""
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda *_: model)

    async def yielding_create_chat_request(session, state):
        await anyio.sleep(0)
        recorded_requests.append(state)

    monkeypatch.setattr("app.chat.stream.create_chat_request", yielding_create_chat_request)

    async with anyio.create_task_group() as tg:

        async def consume_one_then_leave():
            async for _ in stream_chat_events("q"):
                tg.cancel_scope.cancel()

        tg.start_soon(consume_one_then_leave)

    [state] = recorded_requests
    assert state.outcome is ChatOutcome.ABORTED
    assert len(state.sources) == 2


async def test_refused_stream_carries_the_refusal_as_its_answer_and_records_it(
    monkeypatch, recorded_requests
):
    """No context, so no model call: an empty sources event, the refusal as the one text frame,
    done — and the ledger says refused, with nothing spent past retrieval."""

    async def junk_search(session, request):
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", junk_search)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda *_: model)

    events = [event async for event in stream_chat_events("best pizza topping?")]

    assert events == [
        SourcesEvent(data=()),
        TextEvent(data=REFUSAL_ANSWER),
        DoneEvent(),
    ]
    assert model.received == []
    [state] = recorded_requests
    assert state.outcome is ChatOutcome.REFUSED
    assert [result.step for result in state.steps] == [ChatNode.RETRIEVE, ChatNode.REFUSE]
    assert state.sources == ()
    assert state.token_totals() == (None, None)


class ToolCallStreamingModel(RecordingChatModel):
    """Streams tool calls as litellm's real chunks do; the base fake's `_stream` only
    carries content, so an assess call driven through the messages stream would lose them."""

    def _stream(self, messages, *args, **kwargs):
        message = self._generate(messages, *args, **kwargs).generations[0].message
        assert isinstance(message, AIMessage)
        if message.tool_calls:
            for call in message.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": call["name"],
                                "args": json.dumps(call["args"]),
                                "id": call["id"],
                                "index": 0,
                            }
                        ],
                    )
                )
        elif isinstance(message.content, str) and message.content:
            for token in re.split(r"(\s)", message.content):
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        if self.usage:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=self.usage))


class TestLoopStreaming:
    @pytest.fixture
    def one_assess_round(self, monkeypatch):
        assess = ToolCallStreamingModel(
            messages=iter([tool_call_message("search", {"query": "gap"}), AIMessage(content="")]),
            usage=USAGE,
        )
        monkeypatch.setattr("app.chat.graph.assess_model", lambda: assess)

        async def fake_run_tool_call(call):
            return (search_result(id=2, citation="Article 5(1)"),)

        monkeypatch.setattr("app.chat.graph.run_tool_call", fake_run_tool_call)

    async def test_sources_arrive_once_with_the_merged_context(
        self, loop_on, one_result, one_assess_round, monkeypatch
    ):
        monkeypatch.setattr("app.chat.graph.chat_model", lambda *_: fake_chat_model())

        events = [event async for event in stream_chat_events("q")]

        sources_events = [e for e in events if isinstance(e, SourcesEvent)]
        assert len(sources_events) == 1
        assert [source.chunk_id for source in sources_events[0].data] == [1, 2]
        assert events.index(sources_events[0]) < events.index(
            next(e for e in events if isinstance(e, TextEvent))
        )

    async def test_assess_turns_leak_no_text_events(
        self, loop_on, one_result, one_assess_round, monkeypatch
    ):
        monkeypatch.setattr(
            "app.chat.graph.chat_model", lambda *_: fake_chat_model("The answer [1].")
        )

        events = [event async for event in stream_chat_events("q")]

        text = "".join(e.data for e in events if isinstance(e, TextEvent))
        assert text == "The answer [1]."
