"""Rerank client: call contract, ordering, and degradation to the fused order."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.llm import LLMError
from app.retrieval import rerank as rerank_module
from app.retrieval.rerank import _rerank

pytestmark = pytest.mark.anyio


def _response(scores: list[tuple[int, float]]) -> SimpleNamespace:
    return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in scores])


async def test_rerank_orders_indices_best_first(monkeypatch):
    async def fake_arerank(**kwargs):
        return _response([(0, 0.1), (1, 0.9), (2, 0.5)])

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    assert await _rerank("q", ["a", "b", "c"]) == [1, 2, 0]


async def test_rerank_sends_configured_call_kwargs(monkeypatch):
    calls = []

    async def fake_arerank(**kwargs):
        calls.append(kwargs)
        return _response([(0, 1.0)])

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    await _rerank("the query", ["only document"])

    call = calls[0]
    assert call["model"] == "voyage/rerank-2.5"
    assert call["query"] == "the query"
    assert call["documents"] == ["only document"]
    assert call["api_key"] == rerank_module.config.VOYAGE_API_KEY
    assert call["timeout"] == 30
    assert "num_retries" not in call


async def test_rerank_wraps_provider_error_without_leaking_provider_text(monkeypatch):
    provider_message = "connection refused by voyageai.com upstream"

    async def fake_arerank(**kwargs):
        raise openai.APIConnectionError(
            message=provider_message, request=httpx.Request("POST", "http://voyageai.example")
        )

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    with pytest.raises(LLMError) as exc_info:
        await _rerank("q", ["a"])

    assert provider_message not in str(exc_info.value)
    assert exc_info.value.transient is True


@pytest.mark.parametrize(
    "results",
    [
        None,
        [],
        [(0, 0.9), (0, 0.8)],
        [(0, 0.9), (5, 0.8)],
        [(0, 0.9)],
    ],
    ids=["none", "empty", "duplicate", "out-of-range", "subset"],
)
async def test_rerank_raises_on_a_response_that_is_not_a_permutation(monkeypatch, results):
    async def fake_arerank(**kwargs):
        return SimpleNamespace(results=None) if results is None else _response(results)

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    with pytest.raises(LLMError):
        await _rerank("q", ["a", "b"])
