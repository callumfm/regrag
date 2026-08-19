"""Chat stream orchestration: every way a stream ends records one request with what it reached."""

import anyio
import pytest

from app.chat.enums import ChatNode, ChatOutcome
from app.chat.models import ChatToken, DoneEvent, SourcesEvent, TokenEvent
from app.chat.prompts import REFUSAL_ANSWER
from app.chat.service import chat_events
from app.core.llm import LLMError
from tests.chat.conftest import USAGE, fake_chat_model
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


async def test_finished_stream_records_timings_sources_and_usage(
    two_results, monkeypatch, recorded_requests
):
    model = fake_chat_model("Two words [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    async for _ in chat_events("q"):
        pass

    [(state, outcome, ttft_ms, total_ms)] = recorded_requests
    assert state.question == "q"
    assert outcome is ChatOutcome.DONE
    assert state.nodes == (ChatNode.RETRIEVE, ChatNode.SYNTHESIZE)
    assert len(state.sources) == 2
    assert state.usage == USAGE
    assert state.retrieve_ms is not None
    assert ttft_ms is not None
    assert 0 <= state.retrieve_ms <= total_ms


async def test_failed_stream_records_what_it_reached(monkeypatch, recorded_requests):
    async def failing_search(session, request):
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.chat.graph.search", failing_search)

    async for _ in chat_events("q"):
        pass

    [(state, outcome, ttft_ms, _)] = recorded_requests
    assert outcome is ChatOutcome.ERROR
    assert state.nodes == ()
    assert state.retrieve_ms is None
    assert ttft_ms is None
    assert state.sources == ()
    assert state.usage is None


async def test_abandoned_stream_still_records(two_results, monkeypatch, recorded_requests):
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    events = chat_events("q")
    await anext(events)
    await events.aclose()

    [(state, outcome, ttft_ms, _)] = recorded_requests
    assert outcome is ChatOutcome.ABORTED
    assert len(state.sources) == 2
    assert ttft_ms is None


async def test_cancelled_stream_still_records(two_results, monkeypatch, recorded_requests):
    """A client leaving cancels the streaming task mid-stream; the record still lands.
    The fake write yields once, so an unshielded await there would be cancelled, not run."""
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    async def yielding_record_request(state, outcome, ttft_ms, total_ms):
        await anyio.sleep(0)
        recorded_requests.append((state, outcome, ttft_ms, total_ms))

    monkeypatch.setattr("app.chat.service.record_request", yielding_record_request)

    async with anyio.create_task_group() as tg:

        async def consume_one_then_leave():
            async for _ in chat_events("q"):
                tg.cancel_scope.cancel()

        tg.start_soon(consume_one_then_leave)

    [(state, outcome, _, _total)] = recorded_requests
    assert outcome is ChatOutcome.ABORTED
    assert len(state.sources) == 2


async def test_refused_stream_carries_the_refusal_as_its_answer_and_records_it(
    monkeypatch, recorded_requests
):
    """No context, so no model call: an empty sources event, the refusal as the one token,
    done — and the ledger says refused, with nothing spent past retrieval."""

    async def junk_search(session, request):
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", junk_search)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    events = [event async for event in chat_events("best pizza topping?")]

    assert events == [
        SourcesEvent(data=()),
        TokenEvent(data=ChatToken(text=REFUSAL_ANSWER)),
        DoneEvent(),
    ]
    assert model.received == []
    [(state, outcome, ttft_ms, _)] = recorded_requests
    assert outcome is ChatOutcome.REFUSED
    assert state.nodes == (ChatNode.RETRIEVE, ChatNode.REFUSE)
    assert state.sources == ()
    assert state.usage is None
    assert state.retrieve_ms is not None
    assert ttft_ms is not None
