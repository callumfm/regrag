"""Chat SSE endpoint: event ordering, payload shapes, and the error path."""

import json

import httpx
import pytest

from app.core.llm import LLMError
from tests.chat.conftest import THINKING, fake_chat_model, make_result, reasoning_chat_model


def read_events(response: httpx.Response) -> list[tuple[str, dict]]:
    """Parse the SSE body into (event, payload) pairs, ignoring pings."""
    events: list[tuple[str, dict]] = []
    name = None
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:") and name is not None:
            events.append((name, json.loads(line.removeprefix("data:").strip())))
            name = None
    return events


@pytest.fixture
def two_results(monkeypatch):
    async def fake_search(session, request):
        return (make_result(), make_result(id=2, citation="Article 5(1)"))

    monkeypatch.setattr("app.chat.graph.search", fake_search)


def test_stream_orders_sources_then_tokens_then_done(client, two_results, monkeypatch):
    model = fake_chat_model("Two words [1].")
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    with client.stream("POST", "/chat", json={"question": "What is FuelEU?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = read_events(response)

    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    answer = "".join(payload["text"] for name, payload in events if name == "token")
    assert answer == "Two words [1]."


def test_block_list_content_streams_text_without_reasoning(client, two_results, monkeypatch):
    monkeypatch.setattr("app.chat.graph.chat_model", reasoning_chat_model)

    with client.stream("POST", "/chat", json={"question": "q"}) as response:
        events = read_events(response)

    answer = "".join(payload["text"] for name, payload in events if name == "token")
    assert answer == "Ships must comply [1]."
    assert THINKING not in json.dumps(events)


def test_sources_event_binds_markers_to_chunks(client, two_results, monkeypatch):
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    with client.stream("POST", "/chat", json={"question": "q"}) as response:
        events = read_events(response)

    sources = events[0][1]
    assert [source["marker"] for source in sources] == [1, 2]
    assert sources[0]["chunk_id"] == 1
    assert sources[0]["celex"] == "32023R1805"
    assert sources[1]["citation"] == "Article 5(1)"


def test_stream_failure_emits_an_error_event(client, monkeypatch):
    async def failing_search(session, request):
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.chat.graph.search", failing_search)

    with client.stream("POST", "/chat", json={"question": "q"}) as response:
        assert response.status_code == 200
        events = read_events(response)

    [(name, payload)] = events
    assert name == "error"
    assert payload["error"] == "LLMError"
    assert payload["message"] == "embedding call failed"


def test_unexpected_failure_emits_a_generic_error_event(client, monkeypatch):
    async def exploding_search(session, request):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("app.chat.graph.search", exploding_search)

    with client.stream("POST", "/chat", json={"question": "q"}) as response:
        events = read_events(response)

    [(name, payload)] = events
    assert name == "error"
    assert payload["error"] == "InternalError"
    assert payload["message"] == "An unexpected error occurred"
    assert "secret internals" not in json.dumps(payload)


def test_empty_question_is_rejected(client):
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422
