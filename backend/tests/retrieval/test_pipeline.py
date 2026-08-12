import pytest
from litellm.types.rerank import RerankResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.core.llm import LLMError
from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import SearchFilters, SearchResult
from app.retrieval.pipeline import search
from app.retrieval.rerank import rerank_results

pytestmark = pytest.mark.anyio


async def test_an_article_query_returns_that_article(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, "Article 11a")

    assert any(result.citation.startswith("Article 11a") for result in found)


async def test_a_topic_filter_excludes_the_other_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, "greenhouse gas emissions", SearchFilters(topic="mrv"))

    assert {result.topic for result in found} == {"mrv"}


async def test_a_paraphrase_reaches_the_article_that_defines_the_term(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, "monitoring plan submitted to the verifier", limit=5)

    assert any(result.citation.startswith("Article 4") for result in found)


async def test_results_come_back_in_fused_order(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, "greenhouse gas emissions")

    assert [result.score for result in found] == sorted(
        (result.score for result in found), reverse=True
    )
    assert any(result.vector_rank is not None and result.text_rank is not None for result in found)


async def test_a_result_records_which_legs_found_it(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, "Article 11a")

    assert all(result.vector_rank is not None or result.text_rank is not None for result in found)
    assert any(result.text_rank is not None for result in found)


async def test_the_limit_caps_the_results(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert len(await search(db_session, "energy", limit=3)) == 3


async def test_a_query_matching_no_keywords_still_returns_vector_hits(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, "the of and")

    assert found
    assert all(result.text_rank is None for result in found)


async def test_a_search_over_an_empty_corpus_returns_nothing(empty_session: AsyncSession) -> None:
    assert await search(empty_session, "verification period") == ()


async def test_the_candidate_pool_can_be_narrowed_per_call(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await search(db_session, "greenhouse gas emissions", candidates=1, limit=10)

    assert len(found) == 2


async def test_a_provider_failure_surfaces_rather_than_returning_nothing(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fail(texts: list[str], **kwargs: object) -> list[list[float]]:
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.retrieval.pipeline.embed", _fail)

    with pytest.raises(LLMError):
        await search(db_session, "verification period")


async def test_rerank_receives_a_pool_wider_than_the_limit(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    pools = []

    async def _capture(query, results, *, limit):
        pools.append(len(results))
        return results[:limit]

    monkeypatch.setattr("app.retrieval.pipeline.rerank_results", _capture)

    found = await search(db_session, "greenhouse gas emissions", limit=3)

    assert len(found) == 3
    assert 3 < pools[0] <= config.RERANK_POOL


async def test_the_reranked_order_is_what_the_caller_receives(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _reverse(query, results, *, limit):
        return tuple(reversed(results))[:limit]

    monkeypatch.setattr("app.retrieval.pipeline.rerank_results", _reverse)

    found = await search(db_session, "greenhouse gas emissions", limit=5)

    assert [result.score for result in found] == sorted(result.score for result in found)


async def test_disabling_rerank_skips_the_step_and_keeps_the_narrow_pool(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    async def _record(query, results, *, limit):
        calls.append(query)
        return results[:limit]

    monkeypatch.setattr("app.retrieval.pipeline.rerank_results", _record)
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    found = await search(db_session, "greenhouse gas emissions", limit=3)

    assert len(found) == 3
    assert calls == []


async def test_the_real_rerank_path_reorders_the_pool(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[SearchResult, ...]] = []

    async def spying_rerank_results(query, results, *, limit):
        captured.append(results)
        return await rerank_results(query, results, limit=limit)

    async def fake_arerank(**kwargs):
        count = len(kwargs["documents"])
        return RerankResponse(
            results=[{"index": i, "relevance_score": float(i)} for i in reversed(range(count))]
        )

    monkeypatch.setattr("app.retrieval.pipeline.rerank_results", spying_rerank_results)
    monkeypatch.setattr("app.retrieval.rerank.litellm.arerank", fake_arerank)

    found = await search(db_session, "greenhouse gas emissions", limit=5)

    assert found == tuple(reversed(captured[0]))[:5]


async def test_a_limit_above_the_rerank_pool_is_honoured(
    db_session: AsyncSession, corpus: list[DocumentChunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    pools = []

    async def _capture(query, results, *, limit):
        pools.append(len(results))
        return results[:limit]

    monkeypatch.setattr("app.retrieval.pipeline.rerank_results", _capture)

    await search(db_session, "greenhouse gas emissions", limit=config.RERANK_POOL + 10)

    assert pools[0] > config.RERANK_POOL
