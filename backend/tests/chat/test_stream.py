"""Chat stream orchestration: every way a stream ends records one request with what it reached."""

import logging

import anyio
import pytest
from sqlalchemy.exc import OperationalError

from app.chat.enums import ChatNode, ChatOutcome
from app.chat.models import DoneEvent, ErrorEvent, SourcesEvent, TextEvent
from app.chat.prompts import REFUSAL_ANSWER
from app.chat.stream import stream_chat_events
from app.core.llm import LLMError
from tests.chat.conftest import fake_chat_model
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


async def test_finished_stream_records_timings_sources_and_usage(
    two_results, monkeypatch, recorded_requests
):
    model = fake_chat_model("Two words [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    async for _ in stream_chat_events("q"):
        pass

    [state] = recorded_requests
    assert state.question == "q"
    assert state.outcome is ChatOutcome.DONE
    retrieve, synthesize = state.nodes
    assert (retrieve.node, synthesize.node) == (ChatNode.RETRIEVE, ChatNode.SYNTHESIZE)
    assert (retrieve.input_tokens, retrieve.output_tokens) == (None, None)
    assert (synthesize.input_tokens, synthesize.output_tokens) == (1500, 40)
    assert len(state.sources) == 2
    assert state.total_ms is not None
    assert 0 <= sum(result.ms for result in state.nodes) <= state.total_ms
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
    assert state.nodes == ()
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
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    events = stream_chat_events("q")
    await anext(events)
    await events.aclose()

    [state] = recorded_requests
    assert state.outcome is ChatOutcome.ABORTED
    assert len(state.sources) == 2
    assert [result.node for result in state.nodes] == [ChatNode.RETRIEVE]


async def test_failed_write_is_logged_not_raised(two_results, monkeypatch, caplog):
    """The ledger write failing after the answer went out is a log line, not a broken stream."""
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

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
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

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
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    events = [event async for event in stream_chat_events("best pizza topping?")]

    assert events == [
        SourcesEvent(data=()),
        TextEvent(data=REFUSAL_ANSWER),
        DoneEvent(),
    ]
    assert model.received == []
    [state] = recorded_requests
    assert state.outcome is ChatOutcome.REFUSED
    assert [result.node for result in state.nodes] == [ChatNode.RETRIEVE, ChatNode.REFUSE]
    assert state.sources == ()
    assert state.token_totals() == (None, None)
