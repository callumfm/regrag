"""Voyage embedding client: call contract, ordering, and error sanitisation."""

import os
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core import llm
from app.core.llm import EmbedInput, LLMError, embed

pytestmark = pytest.mark.anyio


def _response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(data=[{"embedding": vector} for vector in vectors])


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
    assert call["num_retries"] == 3


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


def test_importing_llm_sets_local_model_cost_map_env_var():
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"
