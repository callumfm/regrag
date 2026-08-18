"""Chat SSE endpoint: event ordering, payload shapes, and the error path."""

import json

import httpx

from app.chat.prompts import REFUSAL_ANSWER
from app.core.llm import LLMError
from tests.chat.conftest import THINKING, fake_chat_model, reasoning_chat_model
from tests.conftest import search_result


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


def test_stream_tells_proxies_and_browsers_not_to_buffer(client, two_results, monkeypatch):
    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    with client.stream("POST", "/chat", json={"question": "q"}) as response:
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"


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
    assert payload["request_id"] == response.headers["X-Request-ID"]
    assert "detail" not in payload


def test_unexpected_failure_emits_a_generic_error_event(client, monkeypatch):
    async def exploding_search(session, request):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("app.chat.graph.search", exploding_search)

    with client.stream("POST", "/chat", json={"question": "q"}) as response:
        events = read_events(response)

    [(name, payload)] = events
    assert name == "error"
    assert payload["error"] == "InternalServerError"
    assert payload["message"] == "An unexpected error occurred"
    assert "secret internals" not in json.dumps(payload)


def test_empty_question_is_rejected(client):
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_the_frames_are_documented_as_an_event_stream(client):
    """The generated client types the frames from the one media type sent, as a union
    it can narrow on the event name."""
    spec = client.get("/openapi.json").json()
    content = spec["paths"]["/chat"]["post"]["responses"]["200"]["content"]
    (media_type,) = content
    assert media_type == "text/event-stream"
    schema = content[media_type]["schema"]
    assert schema["discriminator"]["propertyName"] == "event"
    assert schema["discriminator"]["mapping"]["token"] == "#/components/schemas/TokenEvent"
    token_event = spec["components"]["schemas"]["TokenEvent"]
    assert token_event["properties"]["event"]["const"] == "token"
    assert set(token_event["required"]) == {"event", "data"}


def test_a_refused_question_streams_the_refusal_then_done(client, monkeypatch):
    async def junk_search(session, request):
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    model = fake_chat_model()
    monkeypatch.setattr("app.chat.graph.search", junk_search)
    monkeypatch.setattr("app.chat.graph.chat_model", lambda: model)

    with client.stream("POST", "/chat", json={"question": "best pizza topping?"}) as response:
        events = read_events(response)

    assert events == [
        ("sources", []),
        ("token", {"text": REFUSAL_ANSWER}),
        ("done", {}),
    ]
    assert model.received == []
