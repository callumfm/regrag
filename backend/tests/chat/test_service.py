"""Chat stream orchestration: every way a stream ends records one run with what it reached."""

import pytest

from app.chat.enums import ChatOutcome
from app.chat.service import chat_events
from app.core.llm import LLMError
from tests.chat.conftest import USAGE, fake_chat_model

pytestmark = pytest.mark.anyio


async def test_finished_stream_records_timings_sources_and_usage(
    two_results, monkeypatch, recorded_runs
):
    model = fake_chat_model("Two words [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    async for _ in chat_events("q"):
        pass

    [(stats, outcome)] = recorded_runs
    assert outcome is ChatOutcome.DONE
    assert stats.sources == 2
    assert stats.usage == USAGE
    assert stats.retrieve_ms is not None
    assert stats.ttft_ms is not None
    assert 0 <= stats.retrieve_ms <= stats.ttft_ms <= stats.elapsed_ms()


async def test_failed_stream_records_what_it_reached(monkeypatch, recorded_runs):
    async def failing_search(session, request):
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.chat.graph.search", failing_search)

    async for _ in chat_events("q"):
        pass

    [(stats, outcome)] = recorded_runs
    assert outcome is ChatOutcome.ERROR
    assert stats.retrieve_ms is None
    assert stats.ttft_ms is None
    assert stats.sources == 0
    assert stats.usage is None


async def test_abandoned_stream_still_records(two_results, monkeypatch, recorded_runs):
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    events = chat_events("q")
    await anext(events)
    await events.aclose()

    [(stats, outcome)] = recorded_runs
    assert outcome is ChatOutcome.ABORTED
    assert stats.sources == 2
    assert stats.ttft_ms is None
