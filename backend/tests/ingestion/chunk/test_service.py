"""Chunk persistence: reconciling a document's chunks by content hash."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.references import extract_references
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.service import delete_chunks_outside, keyed, upsert_document_chunks
from app.ingestion.enums import SectionKind
from tests.conftest import chunk

pytestmark = pytest.mark.anyio


async def chunk_rows(session: AsyncSession, ref: str = "32023R1805") -> list[DocumentChunk]:
    return list(
        await session.scalars(
            select(DocumentChunk).where(DocumentChunk.ref == ref).order_by(DocumentChunk.id)
        )
    )


async def test_first_upsert_inserts_every_chunk(db_session: AsyncSession):
    delta = await upsert_document_chunks(
        db_session, "32023R1805", [chunk(), chunk(article="5")], "2026-08-05-aaaaaaa"
    )
    assert (delta.added, delta.removed, delta.unchanged) == (2, 0, 0)
    assert len(await chunk_rows(db_session)) == 2


async def test_repeat_upsert_changes_nothing(db_session: AsyncSession):
    chunks = [chunk(), chunk(article="5")]
    await upsert_document_chunks(db_session, "32023R1805", chunks, "2026-08-05-aaaaaaa")
    before = {row.id for row in await chunk_rows(db_session)}

    delta = await upsert_document_chunks(db_session, "32023R1805", chunks, "2026-08-06-bbbbbbb")

    assert (delta.added, delta.removed, delta.unchanged) == (0, 0, 2)
    assert {row.id for row in await chunk_rows(db_session)} == before


async def test_matched_rows_keep_their_original_corpus_version(db_session: AsyncSession):
    await upsert_document_chunks(db_session, "32023R1805", [chunk()], "2026-08-05-aaaaaaa")
    await upsert_document_chunks(db_session, "32023R1805", [chunk()], "2026-08-06-bbbbbbb")

    assert [row.corpus_version for row in await chunk_rows(db_session)] == ["2026-08-05-aaaaaaa"]


async def test_edited_chunk_is_replaced_not_duplicated(db_session: AsyncSession):
    await upsert_document_chunks(db_session, "32023R1805", [chunk()], "2026-08-05-aaaaaaa")

    delta = await upsert_document_chunks(
        db_session, "32023R1805", [chunk(text="Reworded entirely.")], "2026-08-06-bbbbbbb"
    )

    assert (delta.added, delta.removed, delta.unchanged) == (1, 1, 0)
    rows = await chunk_rows(db_session)
    assert [row.text for row in rows] == ["Reworded entirely."]
    assert rows[0].corpus_version == "2026-08-06-bbbbbbb"


async def test_upsert_touches_only_its_own_document(db_session: AsyncSession):
    await upsert_document_chunks(db_session, "32015R0757", [chunk(ref="32015R0757")], "v1")
    await upsert_document_chunks(db_session, "32023R1805", [chunk()], "v1")

    await upsert_document_chunks(db_session, "32023R1805", [], "v2")

    assert len(await chunk_rows(db_session, "32015R0757")) == 1
    assert await chunk_rows(db_session, "32023R1805") == []


async def test_duplicate_chunks_persist_as_separate_occurrences(db_session: AsyncSession):
    delta = await upsert_document_chunks(
        db_session, "32023R1805", [chunk(), chunk()], "2026-08-05-aaaaaaa"
    )
    assert delta.added == 2
    assert sorted(row.occurrence for row in await chunk_rows(db_session)) == [0, 1]


async def test_chunk_fields_are_mapped_onto_the_row(db_session: AsyncSession):
    await upsert_document_chunks(
        db_session,
        "32023R1805",
        [chunk(heading_path=("Chapter I",), annex=None, part=2, parts=3)],
        "2026-08-05-aaaaaaa",
    )
    row = (await chunk_rows(db_session))[0]
    assert row.topic == "fueleu"
    assert row.citation == "Article 4(1)"
    assert row.heading_path == ["Chapter I"]
    assert (row.part, row.parts) == (2, 3)
    assert row.kind is SectionKind.PARAGRAPH


async def test_references_are_stored_as_json(db_session: AsyncSession):
    text = "As set out in Annex I to this Regulation."
    await upsert_document_chunks(
        db_session,
        "32023R1805",
        [chunk(text=text, references=extract_references(text))],
        "2026-08-05-aaaaaaa",
    )
    row = (await chunk_rows(db_session))[0]
    assert row.references
    assert row.references[0]["annex"] == "I"


async def seed_two_topics(session: AsyncSession) -> None:
    """One fueleu document and one mrv document, a chunk each."""
    await upsert_document_chunks(session, "32023R1805", [chunk()], "v1")
    await upsert_document_chunks(
        session, "32015R0757", [chunk(ref="32015R0757", topic="mrv")], "v1"
    )


async def test_delete_chunks_outside_removes_undiscovered_documents(db_session: AsyncSession):
    await seed_two_topics(db_session)

    removed = await delete_chunks_outside(db_session, ["fueleu"], ["32026R0394"])

    assert removed == 1
    assert await chunk_rows(db_session, "32023R1805") == []


async def test_delete_chunks_outside_keeps_discovered_documents(db_session: AsyncSession):
    await seed_two_topics(db_session)

    assert await delete_chunks_outside(db_session, ["fueleu"], ["32023R1805"]) == 0
    assert len(await chunk_rows(db_session, "32023R1805")) == 1


async def test_delete_chunks_outside_never_touches_another_topic(db_session: AsyncSession):
    await seed_two_topics(db_session)

    await delete_chunks_outside(db_session, ["fueleu"], ["32026R0394"])

    assert len(await chunk_rows(db_session, "32015R0757")) == 1


async def test_delete_chunks_outside_refuses_to_wipe_a_topic_on_empty_discovery(
    db_session: AsyncSession,
):
    await seed_two_topics(db_session)

    assert await delete_chunks_outside(db_session, ["fueleu"], []) == 0
    assert len(await chunk_rows(db_session, "32023R1805")) == 1


async def test_delete_chunks_outside_with_no_topics_is_a_noop(db_session: AsyncSession):
    await seed_two_topics(db_session)

    assert await delete_chunks_outside(db_session, [], ["32026R0394"]) == 0
    assert len(await chunk_rows(db_session, "32023R1805")) == 1


def test_occurrence_counts_up_for_duplicates():
    keys = [(digest, n) for _, digest, n in keyed([chunk(), chunk(), chunk()])]
    assert [n for _, n in keys] == [0, 1, 2]
    assert len({digest for digest, _ in keys}) == 1


def test_distinct_chunks_each_start_at_occurrence_zero():
    keys = [(digest, n) for _, digest, n in keyed([chunk(), chunk(article="5")])]
    assert [n for _, n in keys] == [0, 0]
    assert len({digest for digest, _ in keys}) == 2


def test_keyed_yields_the_original_chunks_in_order():
    chunks = [chunk(), chunk(article="5")]
    assert [c for c, _, _ in keyed(chunks)] == chunks
