"""Widening a hit to its whole section: articles, split sections, and what has neither."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.expand import expand_sections
from app.retrieval.models import CHUNK_COLUMNS, RetrievedChunk

pytestmark = pytest.mark.anyio

NO_LIMIT = 1000
"""More chunks than any test corpus section holds, for the tests not about the cap."""


async def chunk_at(session: AsyncSession, celex: str, citation: str) -> RetrievedChunk:
    """One stored chunk as a caller sees it, so expansion is driven by real rows."""
    stmt = select(*CHUNK_COLUMNS).where(
        DocumentChunk.celex == celex, DocumentChunk.citation == citation
    )
    row = (await session.execute(stmt)).one()
    return RetrievedChunk.model_validate(row)


async def test_expand_sections_reaches_the_paragraph_relevance_cannot(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """Article 4(2) never restates its own subject, so only its article carries it here."""
    hit = await chunk_at(db_session, "32023R1805", "Article 4(1)")

    expanded = await expand_sections(db_session, [hit], limit=NO_LIMIT)

    assert [chunk.citation for chunk in expanded] == [
        "Article 4(1)",
        "Article 4(2)",
        "Article 4(3)",
        "Article 4(4)",
    ]


async def test_expand_sections_widens_each_hit_once(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """Two hits in one article are one article, not two copies of it."""
    first = await chunk_at(db_session, "32023R1805", "Article 4(1)")
    third = await chunk_at(db_session, "32023R1805", "Article 4(3)")

    expanded = await expand_sections(db_session, [first, third], limit=NO_LIMIT)

    assert [chunk.citation for chunk in expanded] == [
        "Article 4(1)",
        "Article 4(2)",
        "Article 4(3)",
        "Article 4(4)",
    ]


async def test_expand_sections_cuts_the_tail_of_the_lowest_ranked_section(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """The cap is a guardrail on context size; what it costs is the end of the last article."""
    fifth = await chunk_at(db_session, "32023R1805", "Article 5(1)")
    fourth = await chunk_at(db_session, "32023R1805", "Article 4(1)")
    whole = await expand_sections(db_session, [fifth, fourth], limit=NO_LIMIT)

    capped = await expand_sections(db_session, [fifth, fourth], limit=len(whole) - 1)

    assert capped == whole[:-1]
    assert capped[-1].article == "4"


async def test_expand_sections_keeps_each_article_at_the_rank_it_was_found(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """Rerank decided the order of the articles; expansion only fills each one in."""
    fifth = await chunk_at(db_session, "32023R1805", "Article 5(1)")
    fourth = await chunk_at(db_session, "32023R1805", "Article 4(1)")

    expanded = await expand_sections(db_session, [fifth, fourth], limit=NO_LIMIT)
    articles = list(dict.fromkeys(chunk.article for chunk in expanded))

    assert articles == ["5", "4"]


async def test_expand_sections_passes_an_unsplit_chunk_outside_an_article_through(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """A whole annex section has nothing to widen to, so it arrives as it came."""
    row = (
        await db_session.execute(
            select(*CHUNK_COLUMNS).where(DocumentChunk.annex.is_not(None), DocumentChunk.parts == 1)
        )
    ).first()
    assert row is not None, "the fixture no longer stores a whole annex chunk"
    chunk = RetrievedChunk.model_validate(row)

    assert await expand_sections(db_session, [chunk], limit=NO_LIMIT) == (chunk,)


async def test_expand_sections_reunites_a_section_split_for_length(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """A table cut in two leaves its halves adrift; one half must bring back the other."""
    row = (
        await db_session.execute(
            select(*CHUNK_COLUMNS)
            .where(
                DocumentChunk.annex.is_not(None), DocumentChunk.parts > 1, DocumentChunk.part > 1
            )
            .order_by(DocumentChunk.position)
        )
    ).first()
    assert row is not None, "the fixture no longer stores a split annex section"
    later_part = RetrievedChunk.model_validate(row)

    expanded = await expand_sections(db_session, [later_part], limit=NO_LIMIT)

    assert len(expanded) == later_part.parts
    assert later_part in expanded


async def test_expand_sections_on_nothing_asks_the_database_nothing(
    empty_session: AsyncSession,
) -> None:
    """No hits means no article keys, so the widening query never runs."""
    assert await expand_sections(empty_session, [], limit=NO_LIMIT) == ()
