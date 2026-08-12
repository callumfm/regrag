import random
from collections.abc import Callable

import pytest
from sqlalchemy import Select, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import EMBED_DIMENSIONS
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from app.retrieval.models import SearchFilters
from app.retrieval.service import ITERATIVE_SCAN, get_article, text_leg, vector_leg
from tests.retrieval.conftest import toy_embed

pytestmark = pytest.mark.anyio

NO_FILTERS = SearchFilters()
INVENTED_CELEX = "39999R9999"
SUPERSEDED = 300
"""Dead vectors hugging the query: enough to exhaust hnsw.ef_search, whose default is 40."""
SURVIVORS = 50
"""Live rows a ring farther out, far more than the limit so the assertion is not about recall."""
WANTED = 5
HUGGING, BEHIND = 0.01, 0.1
SEED = 8
"""Fixed, so the graph this test builds and walks is the same one on every run."""


def random_query(rng: random.Random) -> list[float]:
    """A signed query vector, which the corpus's non-negative vectors all sit far from."""
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBED_DIMENSIONS)]


def nudge(rng: random.Random, query: list[float], scale: float) -> list[float]:
    """The query pushed a set distance in a fresh random direction."""
    return [value + scale * (rng.random() - 0.5) for value in query]


async def leg_ids(session: AsyncSession, leg: Select) -> list[int]:
    """The ids one leg returns in its own rank order, run as the fused query would run it."""
    await session.execute(ITERATIVE_SCAN)
    return [row.id for row in await session.execute(leg)]


async def citations_for(session: AsyncSession, chunk_ids: list[int]) -> list[str]:
    """The citation of each id, in the order the ids were given."""
    stmt = select(DocumentChunk.id, DocumentChunk.citation).where(DocumentChunk.id.in_(chunk_ids))
    citations = {chunk_id: citation for chunk_id, citation in (await session.execute(stmt)).all()}
    return [citations[chunk_id] for chunk_id in chunk_ids]


async def test_the_vector_leg_ranks_the_chunk_whose_text_was_embedded_first(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    target = corpus[0]

    found = await leg_ids(db_session, vector_leg(toy_embed(target.text), NO_FILTERS, limit=5))

    assert found[0] == target.id


async def test_the_vector_leg_respects_its_limit(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(db_session, vector_leg(toy_embed("energy"), NO_FILTERS, limit=3))

    assert len(found) == 3


async def test_the_vector_leg_meets_its_limit_past_dead_tuples(
    db_session: AsyncSession,
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    """Re-ingesting a document leaves old vectors dead in the graph, nearer than its new ones."""
    rng = random.Random(SEED)
    query = random_query(rng)
    vectors = [nudge(rng, query, HUGGING) for _ in range(SUPERSEDED)]
    vectors += [nudge(rng, query, BEHIND) for _ in range(SURVIVORS)]
    rows = [
        make_chunk_row(ingest_run, content_hash=f"{index:064d}", embedding=vector)
        for index, vector in enumerate(vectors)
    ]
    db_session.add_all(rows)
    await db_session.flush()
    for row in rows[:SUPERSEDED]:
        await db_session.delete(row)
    await db_session.flush()
    await db_session.execute(text("SET LOCAL enable_seqscan = off"))

    found = await leg_ids(db_session, vector_leg(query, NO_FILTERS, limit=WANTED))

    assert len(found) == WANTED


async def test_the_vector_leg_skips_chunks_with_no_vector(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    unembedded = corpus[0]
    await db_session.execute(
        update(DocumentChunk).where(DocumentChunk.id == unembedded.id).values(embedding=None)
    )

    found = await leg_ids(db_session, vector_leg(toy_embed("energy"), NO_FILTERS, limit=50))

    assert unembedded.id not in found


async def test_the_keyword_leg_finds_an_article_by_its_citation(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(db_session, text_leg("Article 11a", NO_FILTERS, limit=4))

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

    found_11a = await leg_ids(db_session, text_leg("Article 11a", NO_FILTERS, limit=4))
    assert article_11.id not in found_11a
    assert all(
        citation.startswith("Article 11a")
        for citation in await citations_for(db_session, found_11a)
    )

    found_11 = await leg_ids(db_session, text_leg("Article 11", NO_FILTERS, limit=5))
    assert found_11[0] == article_11.id


async def test_a_query_of_only_stopwords_matches_nothing(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert await leg_ids(db_session, text_leg("the of and", NO_FILTERS, limit=50)) == []


async def test_a_topic_filter_excludes_the_other_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(
        db_session, vector_leg(toy_embed("energy"), SearchFilters(topic="mrv"), limit=50)
    )

    stmt = select(DocumentChunk.topic).where(DocumentChunk.id.in_(found))
    assert set(await db_session.scalars(stmt)) == {"mrv"}


async def test_a_celex_filter_narrows_the_keyword_leg_to_one_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await leg_ids(
        db_session, text_leg("energy", SearchFilters(celex="32023R1805"), limit=50)
    )

    stmt = select(DocumentChunk.celex).where(DocumentChunk.id.in_(found))
    assert set(await db_session.scalars(stmt)) == {"32023R1805"}


async def test_get_article_sorts_a_chapeau_first_and_a_lettered_paragraph_after_its_number(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    """The orderings no fixture act exercises: '2' before '11', '11' before '11a'."""
    for index, paragraph in enumerate(["11a", "2", "11", None]):
        db_session.add(
            make_chunk_row(
                ingest_run,
                celex=INVENTED_CELEX,
                article="7",
                paragraph=paragraph,
                citation=f"Article 7({paragraph})" if paragraph else "Article 7",
                content_hash=f"{index:064d}",
            )
        )
    await db_session.flush()

    found = await get_article(db_session, celex=INVENTED_CELEX, article="7")

    assert [chunk.citation for chunk in found] == [
        "Article 7",
        "Article 7(2)",
        "Article 7(11)",
        "Article 7(11a)",
    ]


async def test_get_article_returns_paragraphs_in_reading_order_not_text_order(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await get_article(db_session, celex="32023R1805", article="5")

    assert [chunk.citation for chunk in found] == [f"Article 5({n})" for n in range(1, 11)]


async def test_get_article_puts_a_split_chapeau_in_part_order(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await get_article(db_session, celex="32015R0757", article="3")

    assert [chunk.citation for chunk in found] == ["Article 3", "Article 3"]
    assert len(found) == 2


async def test_get_article_matches_the_article_number_case_insensitively(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert await get_article(db_session, celex="32015R0757", article="11A") == await get_article(
        db_session, celex="32015R0757", article="11a"
    )


async def test_get_article_does_not_reach_into_another_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await get_article(db_session, celex="32015R0757", article="4")

    assert {chunk.celex for chunk in found} == {"32015R0757"}


async def test_an_unknown_article_returns_nothing_rather_than_raising(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert await get_article(db_session, celex="32015R0757", article="999") == ()
