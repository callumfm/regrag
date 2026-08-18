"""Chat stream stats: the one log line written when a stream ends."""

from typing import Any

import pytest

from app.chat import stats
from app.chat.service import chat_events
from app.core.llm import LLMError
from tests.chat.conftest import USAGE, fake_chat_model
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


@pytest.fixture
def stats_lines(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    """Capture (rendered message, extra) for each stats line the service logs."""
    lines: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        stats.logger, "info", lambda msg, *args, extra: lines.append((msg % args, extra))
    )
    return lines


@pytest.fixture
def two_results(monkeypatch):
    async def fake_search(session, request):
        return (search_result(), search_result(id=2, citation="Article 5(1)"))

    monkeypatch.setattr("app.chat.graph.search", fake_search)


async def test_finished_stream_logs_timings_sources_and_usage(
    two_results, monkeypatch, stats_lines
):
    model = fake_chat_model("Two words [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    async for _ in chat_events("q"):
        pass

    [(message, extra)] = stats_lines
    assert message.startswith("chat done")
    assert extra["outcome"] == "done"
    assert extra["sources"] == 2
    assert extra["input_tokens"] == USAGE["input_tokens"]
    assert extra["output_tokens"] == USAGE["output_tokens"]
    assert 0 <= extra["retrieve_ms"] <= extra["ttft_ms"] <= extra["total_ms"]


async def test_failed_stream_logs_what_it_reached(monkeypatch, stats_lines):
    async def failing_search(session, request):
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.chat.graph.search", failing_search)

    async for _ in chat_events("q"):
        pass

    [(message, extra)] = stats_lines
    assert extra["outcome"] == "error"
    assert extra["retrieve_ms"] is None
    assert extra["ttft_ms"] is None
    assert extra["sources"] == 0
    assert extra["input_tokens"] is None
    assert extra["total_ms"] >= 0


async def test_abandoned_stream_still_logs(two_results, monkeypatch, stats_lines):
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    events = chat_events("q")
    await anext(events)
    await events.aclose()

    [(_, extra)] = stats_lines
    assert extra["outcome"] == "aborted"
    assert extra["sources"] == 2
    assert extra["ttft_ms"] is None
