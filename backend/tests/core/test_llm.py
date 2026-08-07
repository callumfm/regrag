"""Voyage embedding client: call contract, ordering, and error sanitisation."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core import llm
from app.core.llm import TRANSIENT_PROVIDER_ERRORS, EmbedInput, LLMError, embed

pytestmark = pytest.mark.anyio

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(
        data=[{"embedding": vector, "index": i} for i, vector in enumerate(vectors)]
    )


def test_embed_input_document_value():
    assert EmbedInput.DOCUMENT == "document"


def test_embed_input_query_value():
    assert EmbedInput.QUERY == "query"


def test_llm_error_status_code():
    assert LLMError("x").status_code == 502


async def test_embed_empty_input_returns_empty_without_calling_provider(monkeypatch):
    calls = []

    async def fake_aembedding(**kwargs):
        calls.append(kwargs)
        return _response([])

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    result = await embed([], input_type=EmbedInput.DOCUMENT)

    assert result == []
    assert calls == []


async def test_embed_returns_vectors_in_input_order(monkeypatch):
    async def fake_aembedding(**kwargs):
        return _response([[float(i)] for i in range(len(kwargs["input"]))])

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    result = await embed(["a", "b", "c"], input_type=EmbedInput.DOCUMENT)

    assert result == [[0.0], [1.0], [2.0]]


async def test_embed_reorders_shuffled_response_by_index(monkeypatch):
    async def fake_aembedding(**kwargs):
        n = len(kwargs["input"])
        data = [{"embedding": [float(i)], "index": i} for i in reversed(range(n))]
        return SimpleNamespace(data=data)

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    result = await embed(["a", "b", "c"], input_type=EmbedInput.DOCUMENT)

    assert result == [[0.0], [1.0], [2.0]]


async def test_embed_raises_on_short_response(monkeypatch):
    async def fake_aembedding(**kwargs):
        return _response([[0.0]])

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    with pytest.raises(LLMError):
        await embed(["a", "b", "c"], input_type=EmbedInput.DOCUMENT)


async def test_embed_sends_configured_call_kwargs(monkeypatch):
    calls = []

    async def fake_aembedding(**kwargs):
        calls.append(kwargs)
        return _response([[0.0]])

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    await embed(["text"], input_type=EmbedInput.DOCUMENT)

    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "voyage/voyage-4-lite"
    assert call["dimensions"] == 1024
    assert call["api_key"] == llm.config.VOYAGE_API_KEY
    assert call["timeout"] == 30
    assert "num_retries" not in call


@pytest.mark.parametrize(
    ("input_type", "expected"),
    [
        (EmbedInput.DOCUMENT, "document"),
        (EmbedInput.QUERY, "query"),
    ],
)
async def test_embed_sends_input_type(monkeypatch, input_type, expected):
    calls = []

    async def fake_aembedding(**kwargs):
        calls.append(kwargs)
        return _response([[0.0]])

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    await embed(["text"], input_type=input_type)

    assert calls[0]["input_type"] == expected


async def test_embed_wraps_provider_error_without_leaking_provider_text(monkeypatch):
    provider_message = "connection refused by voyageai.com upstream"

    async def fake_aembedding(**kwargs):
        raise openai.APIConnectionError(
            message=provider_message, request=httpx.Request("POST", "http://voyageai.example")
        )

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    with pytest.raises(LLMError) as exc_info:
        await embed(["text"], input_type=EmbedInput.DOCUMENT)

    assert provider_message not in str(exc_info.value)


def test_importing_llm_without_cost_map_pin_makes_no_network_connection():
    """Regression guard for the offline pin: unset it and prove no socket connect happens."""
    script = """
import os
import socket

os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)

attempts = []


def _tracking_connect(self, address):
    attempts.append(address)
    raise OSError("blocked for offline test")


socket.socket.connect = _tracking_connect

import app.core.llm  # noqa: E402

print(len(attempts))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0", result.stdout + result.stderr


async def test_embed_sends_voyage_request_body_and_auth_header(monkeypatch):
    """One integration-shaped check of the real request litellm builds for Voyage."""
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        n = len(captured["body"]["input"])
        payload = {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": [0.1]} for i in range(n)],
            "model": captured["body"]["model"],
            "usage": {"total_tokens": 7},
        }
        return httpx.Response(200, json=payload, request=request)

    transport = httpx.MockTransport(handler)
    original_init = AsyncHTTPHandler.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.client = httpx.AsyncClient(transport=transport)

    monkeypatch.setattr(AsyncHTTPHandler, "__init__", patched_init)
    llm.litellm.in_memory_llm_clients_cache.flush_cache()

    result = await embed(["alpha", "beta"], input_type=EmbedInput.QUERY)

    assert result == [[0.1], [0.1]]
    assert captured["body"]["input"] == ["alpha", "beta"]
    assert captured["body"]["model"] == "voyage-4-lite"
    assert captured["body"]["output_dimension"] == 1024
    assert captured["body"]["input_type"] == "query"
    assert "num_retries" not in captured["body"]
    assert captured["headers"]["authorization"] == f"Bearer {llm.config.VOYAGE_API_KEY}"

    llm.litellm.in_memory_llm_clients_cache.flush_cache()


REQUEST = httpx.Request("POST", "https://api.voyageai.com/v1/embeddings")


def provider_error(exc_type, status_code: int | None = None):
    """Build an openai exception the way litellm surfaces one."""
    if status_code is None:
        return exc_type(request=REQUEST)
    return exc_type(
        message="provider said no",
        response=httpx.Response(status_code, request=REQUEST),
        body=None,
    )


def test_llm_error_is_not_transient_by_default():
    assert LLMError("nope").transient is False


def test_llm_error_records_transience_when_told():
    assert LLMError("nope", transient=True).transient is True


@pytest.mark.parametrize(
    ("exc_type", "status_code"),
    [
        (openai.RateLimitError, 429),
        (openai.InternalServerError, 500),
        (openai.APITimeoutError, None),
        (openai.APIConnectionError, None),
    ],
)
async def test_retryable_provider_failures_are_flagged_transient(
    monkeypatch, exc_type, status_code
):
    async def fake_aembedding(**kwargs):
        raise provider_error(exc_type, status_code)

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    with pytest.raises(LLMError) as caught:
        await embed(["a chunk"], input_type=EmbedInput.DOCUMENT)
    assert caught.value.transient is True


@pytest.mark.parametrize(
    ("exc_type", "status_code"),
    [
        (openai.AuthenticationError, 401),
        (openai.BadRequestError, 400),
        (openai.NotFoundError, 404),
    ],
)
async def test_client_errors_are_never_flagged_transient(monkeypatch, exc_type, status_code):
    async def fake_aembedding(**kwargs):
        raise provider_error(exc_type, status_code)

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    with pytest.raises(LLMError) as caught:
        await embed(["a chunk"], input_type=EmbedInput.DOCUMENT)
    assert caught.value.transient is False


async def test_a_misaligned_response_is_not_transient(monkeypatch):
    async def fake_aembedding(**kwargs):
        return _response([[1.0]])

    monkeypatch.setattr(llm.litellm, "aembedding", fake_aembedding)

    with pytest.raises(LLMError) as caught:
        await embed(["a", "b"], input_type=EmbedInput.DOCUMENT)
    assert caught.value.transient is False


def test_transient_errors_exclude_the_permanent_ones():
    assert openai.RateLimitError in TRANSIENT_PROVIDER_ERRORS
    assert openai.AuthenticationError not in TRANSIENT_PROVIDER_ERRORS
    assert openai.BadRequestError not in TRANSIENT_PROVIDER_ERRORS
