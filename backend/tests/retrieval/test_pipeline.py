import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import LLMError
from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import SearchFilters
from app.retrieval.pipeline import search

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
