from collections.abc import Callable

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.models import Reference
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from app.retrieval.models import ReferenceTarget, RetrievedChunk
from app.retrieval.related import CHUNK_COLUMNS, expand_sections, follow_reference

pytestmark = pytest.mark.anyio

INVENTED_CELEX = "39999R9999"


async def stored_reference(
    session: AsyncSession, *, celex: str, predicate: Callable[[Reference], bool]
) -> Reference:
    """One cross-reference the chunker really stored, so the follow is driven by real data."""
    stmt = select(DocumentChunk.references).where(DocumentChunk.celex == celex)
    found = [
        reference
        for (references,) in await session.execute(stmt)
        for reference in map(Reference.model_validate, references)
        if predicate(reference)
    ]
    assert found, f"the {celex} fixture no longer stores a reference this test can follow"
    return found[0]


async def test_follow_reference_sorts_a_chapeau_first_and_a_lettered_paragraph_after_its_number(
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

    found = await follow_reference(db_session, ReferenceTarget(celex=INVENTED_CELEX, article="7"))

    assert [chunk.citation for chunk in found] == [
        "Article 7",
        "Article 7(2)",
        "Article 7(11)",
        "Article 7(11a)",
    ]


async def test_follow_reference_returns_paragraphs_in_reading_order_not_text_order(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await follow_reference(db_session, ReferenceTarget(celex="32023R1805", article="5"))

    assert [chunk.citation for chunk in found] == [f"Article 5({n})" for n in range(1, 11)]


async def test_follow_reference_puts_a_split_chapeau_in_part_order(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """A RetrievedChunk carries no part, so the order is checked against the rows' own parts."""
    found = await follow_reference(db_session, ReferenceTarget(celex="32015R0757", article="3"))

    parts = select(DocumentChunk.id, DocumentChunk.part).where(
        DocumentChunk.celex == "32015R0757", DocumentChunk.article == "3"
    )
    part_of = {id_: part for id_, part in await db_session.execute(parts)}
    assert [chunk.citation for chunk in found] == ["Article 3", "Article 3"]
    assert [part_of[chunk.id] for chunk in found] == [1, 2]


async def test_follow_reference_matches_the_article_number_case_insensitively(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    upper = await follow_reference(db_session, ReferenceTarget(celex="32015R0757", article="11A"))
    lower = await follow_reference(db_session, ReferenceTarget(celex="32015R0757", article="11a"))

    assert lower
    assert upper == lower


async def test_follow_reference_does_not_reach_into_another_act(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await follow_reference(db_session, ReferenceTarget(celex="32015R0757", article="4"))

    assert {chunk.celex for chunk in found} == {"32015R0757"}


async def test_an_unknown_article_returns_nothing_rather_than_raising(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    assert (
        await follow_reference(db_session, ReferenceTarget(celex="32015R0757", article="999")) == ()
    )


async def test_an_act_outside_the_corpus_returns_nothing_rather_than_raising(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """The agent has to be able to say the text is not held, which an exception denies it."""
    assert (
        await follow_reference(db_session, ReferenceTarget(celex=INVENTED_CELEX, article="1")) == ()
    )


async def test_a_paragraph_narrows_to_the_paragraph_cited(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    found = await follow_reference(
        db_session, ReferenceTarget(celex="32023R1805", article="5", paragraph="7")
    )

    assert [chunk.citation for chunk in found] == ["Article 5(7)"]


async def test_an_annex_comes_back_in_document_order(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """An annex has no paragraph to sort on, so position is the only reading order it has."""
    found = await follow_reference(db_session, ReferenceTarget(celex="32015R0757", annex="I"))

    positions = select(DocumentChunk.id, DocumentChunk.position).where(
        DocumentChunk.celex == "32015R0757", DocumentChunk.annex == "I"
    )
    order = {id_: position for id_, position in await db_session.execute(positions)}
    assert len(found) == 5
    assert [order[chunk.id] for chunk in found] == sorted(order.values())


async def test_an_annex_a_single_annex_act_left_unnumbered_is_still_addressable(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    """RRG-75 writes '' for an act whose one annex carries no number; None is not-in-an-annex."""
    db_session.add(
        make_chunk_row(
            ingest_run,
            celex=INVENTED_CELEX,
            article=None,
            annex="",
            citation="Annex",
            content_hash=f"{1:064d}",
        )
    )
    await db_session.flush()

    found = await follow_reference(db_session, ReferenceTarget(celex=INVENTED_CELEX, annex=""))

    assert [chunk.citation for chunk in found] == ["Annex"]


async def test_following_a_stored_reference_reaches_the_cited_text(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """RRG-58's first acceptance, driven by a reference the chunker really stored."""
    reference = await stored_reference(
        db_session, celex="32023R1805", predicate=lambda r: r.article == "4"
    )

    found = await follow_reference(
        db_session, ReferenceTarget.from_reference(reference, citing="32023R1805")
    )

    assert [chunk.citation for chunk in found] == ["Article 4(1)"]
    assert found[0].text


async def test_following_a_stored_reference_to_an_act_outside_the_corpus_is_empty(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """RRG-58's second acceptance: the corpus holds 32023R1805 but not its Article 10."""
    reference = await stored_reference(
        db_session,
        celex="32015R0757",
        predicate=lambda r: r.instrument == "32023R1805" and r.article is not None,
    )

    found = await follow_reference(
        db_session, ReferenceTarget.from_reference(reference, citing="32015R0757")
    )

    assert found == ()


async def test_a_followed_chunk_carries_the_reference_that_leads_on_from_it(
    db_session: AsyncSession, corpus: list[DocumentChunk]
) -> None:
    """The second hop reads the link off the chunk, rather than parsing it back out of the prose."""
    first = await follow_reference(
        db_session, ReferenceTarget(celex="32015R0757", article="4", paragraph="8")
    )

    (cited,) = [reference for chunk in first for reference in chunk.references]

    assert cited.raw == "Article 11a"
    assert cited.instrument is None
    second = await follow_reference(
        db_session, ReferenceTarget.from_reference(cited, citing="32015R0757")
    )
    assert second
    assert all(chunk.citation.startswith("Article 11a") for chunk in second)


async def test_a_chunk_citing_nothing_carries_no_references(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
    ingest_run: IngestRun,
    make_chunk_row: Callable[..., DocumentChunk],
) -> None:
    """An empty column reads as no links, not as a missing field the caller has to guard."""
    db_session.add(
        make_chunk_row(
            ingest_run,
            celex=INVENTED_CELEX,
            article="2",
            citation="Article 2",
            content_hash=f"{2:064d}",
            references=[],
        )
    )
    await db_session.flush()

    found = await follow_reference(db_session, ReferenceTarget(celex=INVENTED_CELEX, article="2"))

    assert [chunk.references for chunk in found] == [()]


async def chunk_at(session: AsyncSession, celex: str, citation: str) -> RetrievedChunk:
    """One stored chunk as a caller sees it, so expansion is driven by real rows."""
    stmt = select(*CHUNK_COLUMNS).where(
        DocumentChunk.celex == celex, DocumentChunk.citation == citation
    )
    row = (await session.execute(stmt)).one()
    return RetrievedChunk.model_validate(row, from_attributes=True)


async def test_expand_sections_reaches_the_paragraph_relevance_cannot(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """Article 4(2) never restates its own subject, so only its article carries it here."""
    hit = await chunk_at(db_session, "32023R1805", "Article 4(1)")

    expanded = await expand_sections(db_session, [hit])

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

    expanded = await expand_sections(db_session, [first, third])

    assert [chunk.citation for chunk in expanded] == [
        "Article 4(1)",
        "Article 4(2)",
        "Article 4(3)",
        "Article 4(4)",
    ]


async def test_expand_sections_keeps_each_article_at_the_rank_it_was_found(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """Rerank decided the order of the articles; expansion only fills each one in."""
    fifth = await chunk_at(db_session, "32023R1805", "Article 5(1)")
    fourth = await chunk_at(db_session, "32023R1805", "Article 4(1)")

    expanded = await expand_sections(db_session, [fifth, fourth])
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
    chunk = RetrievedChunk.model_validate(row, from_attributes=True)

    assert await expand_sections(db_session, [chunk]) == (chunk,)


async def test_expand_sections_reunites_a_section_split_for_length(
    db_session: AsyncSession,
    corpus: list[DocumentChunk],
) -> None:
    """A table cut in two leaves its halves adrift; one half must bring back the other."""
    row = (
        await db_session.execute(
            select(*CHUNK_COLUMNS, DocumentChunk.parts)
            .where(
                DocumentChunk.annex.is_not(None), DocumentChunk.parts > 1, DocumentChunk.part > 1
            )
            .order_by(DocumentChunk.position)
        )
    ).first()
    assert row is not None, "the fixture no longer stores a split annex section"
    later_part = RetrievedChunk.model_validate(row, from_attributes=True)

    expanded = await expand_sections(db_session, [later_part])

    assert len(expanded) == row.parts
    assert later_part in expanded


async def test_expand_sections_on_nothing_asks_the_database_nothing(
    empty_session: AsyncSession,
) -> None:
    """No hits means no article keys, so the widening query never runs."""
    assert await expand_sections(empty_session, []) == ()
