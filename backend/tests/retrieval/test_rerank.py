"""Rerank client: call contract, ordering, and degradation to the fused order."""

import httpx
import openai
import pytest
from litellm.types.rerank import RerankResponse
from tenacity import stop_after_attempt

from app.core.llm import LLMError
from app.retrieval import rerank as rerank_module
from app.retrieval.models import SearchResult
from app.retrieval.rerank import _rerank, rerank_results
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


def _response(scores: list[tuple[int, float]]) -> RerankResponse:
    return RerankResponse(results=[{"index": i, "relevance_score": s} for i, s in scores])


def _result(chunk_id: int) -> SearchResult:
    """The chunk_id-th hit of a fused ranking, its score falling with its rank."""
    return search_result(
        id=chunk_id,
        citation=f"Article {chunk_id}",
        article=str(chunk_id),
        text=f"text of chunk {chunk_id}",
        position=chunk_id,
        rrf_score=1.0 / chunk_id,
        vector_rank=chunk_id,
        text_rank=None,
    )


async def test_rerank_returns_the_provider_ranking_in_order(monkeypatch):
    async def fake_arerank(**kwargs):
        return _response([(1, 0.9), (2, 0.5), (0, 0.1)])

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    ranked = await _rerank("q", ["a", "b", "c"])

    assert [item["index"] for item in ranked] == [1, 2, 0]


async def test_rerank_sends_configured_call_kwargs(monkeypatch):
    calls = []

    async def fake_arerank(**kwargs):
        calls.append(kwargs)
        return _response([(0, 1.0)])

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    await _rerank("the query", ["only document"])

    call = calls[0]
    assert call["model"] == rerank_module.config.RERANK_MODEL
    assert call["query"] == "the query"
    assert call["documents"] == ["only document"]
    assert call["api_key"] == rerank_module.config.VOYAGE_API_KEY
    assert call["timeout"] == rerank_module.config.RERANK_TIMEOUT
    assert "num_retries" not in call


async def test_rerank_wraps_provider_error_without_leaking_provider_text(monkeypatch):
    provider_message = "connection refused by voyageai.com upstream"

    async def fake_arerank(**kwargs):
        raise openai.APIConnectionError(
            message=provider_message, request=httpx.Request("POST", "http://voyageai.example")
        )

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    with pytest.raises(LLMError) as exc_info:
        await _rerank.retry_with(stop=stop_after_attempt(1))("q", ["a"])

    assert provider_message not in str(exc_info.value)
    assert exc_info.value.transient is True


async def test_a_null_results_field_leaves_the_fused_order(monkeypatch):
    async def fake_arerank(**kwargs):
        return RerankResponse(results=None)

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    assert await _rerank("q", ["a", "b"]) == []


async def test_rerank_results_reorders_and_cuts_to_the_limit(monkeypatch):
    async def fake_arerank(**kwargs):
        return _response([(2, 0.9), (0, 0.5), (1, 0.1)])

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)
    fused = (_result(1), _result(2), _result(3))

    reranked = await rerank_results("q", fused, limit=2)

    assert [result.id for result in reranked] == [3, 1]


async def test_rerank_results_sends_the_chunk_texts(monkeypatch):
    calls = []

    async def fake_arerank(**kwargs):
        calls.append(kwargs)
        return _response([(0, 0.9), (1, 0.1)])

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    await rerank_results("q", (_result(1), _result(2)), limit=2)

    assert calls[0]["documents"] == ["text of chunk 1", "text of chunk 2"]


async def test_a_provider_failure_degrades_to_the_fused_order(monkeypatch):
    async def fake_arerank(**kwargs):
        raise openai.OpenAIError("provider rejected the request")

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)
    fused = (_result(1), _result(2), _result(3))

    assert await rerank_results("q", fused, limit=2) == fused[:2]


async def test_empty_results_return_empty_without_calling_the_provider(monkeypatch):
    calls = []

    async def fake_arerank(**kwargs):
        calls.append(kwargs)
        return _response([])

    monkeypatch.setattr(rerank_module.litellm, "arerank", fake_arerank)

    assert await rerank_results("q", (), limit=10) == ()
    assert calls == []
