from collections.abc import Callable

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from app.retrieval.models import SearchFilters
from app.retrieval.service import text_search, vector_search
from tests.retrieval.conftest import toy_embed

pytestmark = pytest.mark.anyio

NO_FILTERS = SearchFilters()


async def citations_for(session: AsyncSession, chunk_ids: list[int]) -> list[str]:
    """The citation of each id, in the order the ids were given."""
    stmt = select(DocumentChunk.id, DocumentChunk.citation).where(DocumentChunk.id.in_(chunk_ids))
    citations = {chunk_id: citation for chunk_id, citation in (await session.execute(stmt)).all()}
    return [citations[chunk_id] for chunk_id in chunk_ids]


async def test_the_vector_leg_ranks_the_chunk_whose_text_was_embedded_first(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    target = corpus[0]

    found = await vector_search(db_session, toy_embed(target.text), NO_FILTERS, limit=5)

    assert found[0] == target.id


async def test_the_vector_leg_respects_its_limit(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await vector_search(db_session, toy_embed("energy"), NO_FILTERS, limit=3)

    assert len(found) == 3


async def test_the_vector_leg_skips_chunks_with_no_vector(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    unembedded = corpus[0]
    unembedded.embedding = None
    await db_session.flush()

    found = await vector_search(db_session, toy_embed("energy"), NO_FILTERS, limit=50)

    assert unembedded.id not in found


async def test_the_keyword_leg_finds_an_article_by_its_citation(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await text_search(db_session, "Article 11a", NO_FILTERS, limit=4)

    assert found
    assert all(
        citation.startswith("Article 11a") for citation in await citations_for(db_session, found)
    )


async def test_the_keyword_leg_does_not_confuse_article_11_with_article_11a(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    article_11 = make_chunk_row(
        ingest_run,
        content_hash="c" * 64,
        article="11",
        paragraph="1",
        citation="Article 11(1)",
        text="Companies shall submit their monitoring plan to the verifier before the period.",
    )
    db_session.add(article_11)
    await db_session.flush()

    found_11a = await text_search(db_session, "Article 11a", NO_FILTERS, limit=4)
    assert article_11.id not in found_11a
    assert all(
        citation.startswith("Article 11a")
        for citation in await citations_for(db_session, found_11a)
    )

    found_11 = await text_search(db_session, "Article 11", NO_FILTERS, limit=5)
    assert found_11[0] == article_11.id


async def test_a_query_of_only_stopwords_matches_nothing(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert await text_search(db_session, "the of and", NO_FILTERS, limit=50) == []


async def test_a_topic_filter_excludes_the_other_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await vector_search(
        db_session, toy_embed("energy"), SearchFilters(topic="mrv"), limit=50
    )

    stmt = select(DocumentChunk.topic).where(DocumentChunk.id.in_(found))
    assert set(await db_session.scalars(stmt)) == {"mrv"}


async def test_a_celex_filter_narrows_the_keyword_leg_to_one_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await text_search(db_session, "energy", SearchFilters(celex="32023R1805"), limit=50)

    stmt = select(DocumentChunk.celex).where(DocumentChunk.id.in_(found))
    assert set(await db_session.scalars(stmt)) == {"32023R1805"}
