"""Chunk persistence: reconciling a document's chunks by content hash."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.ingestion.chunk.references import extract_references
from app.ingestion.chunk.service import (
    count_embedded_chunks,
    delete_chunks_outside,
    get_unembedded_chunks,
    keyed,
    upsert_document_chunks,
)
from app.ingestion.enums import SectionKind
from app.ingestion.schemas import IngestRun
from tests.conftest import chunk, chunk_rows

pytestmark = pytest.mark.anyio

VECTOR = [0.1] * config.EMBED_DIMENSIONS


async def test_first_upsert_inserts_every_chunk(db_session: AsyncSession, ingest_run: IngestRun):
    result = await upsert_document_chunks(
        db_session,
        celex="32023R1805",
        chunks=[chunk(), chunk(article="5")],
        ingest_run_id=ingest_run.id,
    )
    assert (result.added, result.removed, result.unchanged) == (2, 0, 0)
    assert len(await chunk_rows(db_session)) == 2


async def test_repeat_upsert_changes_nothing(db_session: AsyncSession, ingest_run: IngestRun):
    chunks = [chunk(), chunk(article="5")]
    await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=chunks, ingest_run_id=ingest_run.id
    )
    before = {row.id for row in await chunk_rows(db_session)}

    result = await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=chunks, ingest_run_id=ingest_run.id
    )

    assert (result.added, result.removed, result.unchanged) == (0, 0, 2)
    assert {row.id for row in await chunk_rows(db_session)} == before


async def test_matched_rows_keep_the_run_they_first_appeared_in(
    db_session: AsyncSession, ingest_run: IngestRun, later_run: IngestRun
):
    await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=[chunk()], ingest_run_id=ingest_run.id
    )
    await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=[chunk()], ingest_run_id=later_run.id
    )

    assert [row.ingest_run_id for row in await chunk_rows(db_session)] == [ingest_run.id]


async def test_edited_chunk_is_replaced_not_duplicated(
    db_session: AsyncSession, ingest_run: IngestRun, later_run: IngestRun
):
    await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=[chunk()], ingest_run_id=ingest_run.id
    )

    result = await upsert_document_chunks(
        db_session,
        celex="32023R1805",
        chunks=[chunk(text="Reworded entirely.")],
        ingest_run_id=later_run.id,
    )

    assert (result.added, result.removed, result.unchanged) == (1, 1, 0)
    rows = await chunk_rows(db_session)
    assert [row.text for row in rows] == ["Reworded entirely."]
    assert rows[0].ingest_run_id == later_run.id


async def test_upsert_touches_only_its_own_document(
    db_session: AsyncSession, ingest_run: IngestRun
):
    await upsert_document_chunks(
        db_session,
        celex="32015R0757",
        chunks=[chunk(celex="32015R0757")],
        ingest_run_id=ingest_run.id,
    )
    await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=[chunk()], ingest_run_id=ingest_run.id
    )

    await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=[], ingest_run_id=ingest_run.id
    )

    assert len(await chunk_rows(db_session, "32015R0757")) == 1
    assert await chunk_rows(db_session, "32023R1805") == []


async def test_duplicate_chunks_persist_as_separate_occurrences(
    db_session: AsyncSession, ingest_run: IngestRun
):
    result = await upsert_document_chunks(
        db_session, celex="32023R1805", chunks=[chunk(), chunk()], ingest_run_id=ingest_run.id
    )
    assert result.added == 2
    assert sorted(row.occurrence for row in await chunk_rows(db_session)) == [0, 1]


async def test_chunk_fields_are_mapped_onto_the_row(
    db_session: AsyncSession, ingest_run: IngestRun
):
    await upsert_document_chunks(
        db_session,
        celex="32023R1805",
        chunks=[chunk(heading_path=("Chapter I",), annex=None, part=2, parts=3)],
        ingest_run_id=ingest_run.id,
    )
    row = (await chunk_rows(db_session))[0]
    assert row.topic == "fueleu"
    assert row.citation == "Article 4(1)"
    assert row.heading_path == ["Chapter I"]
    assert (row.part, row.parts) == (2, 3)
    assert row.kind is SectionKind.PARAGRAPH


async def test_references_are_stored_as_json(db_session: AsyncSession, ingest_run: IngestRun):
    text = "As set out in Annex I to this Regulation."
    await upsert_document_chunks(
        db_session,
        celex="32023R1805",
        chunks=[chunk(text=text, references=extract_references(text))],
        ingest_run_id=ingest_run.id,
    )
    row = (await chunk_rows(db_session))[0]
    assert row.references
    assert row.references[0]["annex"] == "I"


async def seed_two_topics(session: AsyncSession, run: IngestRun) -> None:
    """One fueleu document and one mrv document, a chunk each."""
    await upsert_document_chunks(
        session, celex="32023R1805", chunks=[chunk()], ingest_run_id=run.id
    )
    await upsert_document_chunks(
        session,
        celex="32015R0757",
        chunks=[chunk(celex="32015R0757", topic="mrv")],
        ingest_run_id=run.id,
    )


async def test_delete_chunks_outside_removes_documents_off_the_keep_list(
    db_session: AsyncSession, ingest_run: IngestRun
):
    await seed_two_topics(db_session, ingest_run)

    removed = await delete_chunks_outside(db_session, corpus_celexes=["32015R0757"])

    assert removed == 1
    assert await chunk_rows(db_session, "32023R1805") == []


async def test_delete_chunks_outside_keeps_documents_on_the_keep_list(
    db_session: AsyncSession, ingest_run: IngestRun
):
    await seed_two_topics(db_session, ingest_run)

    assert await delete_chunks_outside(db_session, corpus_celexes=["32023R1805", "32015R0757"]) == 0
    assert len(await chunk_rows(db_session, "32023R1805")) == 1


async def test_delete_chunks_outside_spares_a_celex_another_topic_still_holds(
    db_session: AsyncSession, ingest_run: IngestRun
):
    """The corpus, not the topic tag, decides: a shared celex survives on any topic's say-so."""
    await seed_two_topics(db_session, ingest_run)

    await delete_chunks_outside(db_session, corpus_celexes=["32015R0757"])

    assert len(await chunk_rows(db_session, "32015R0757")) == 1


async def test_delete_chunks_outside_refuses_to_wipe_everything_on_an_empty_keep_list(
    db_session: AsyncSession, ingest_run: IngestRun
):
    await seed_two_topics(db_session, ingest_run)

    assert await delete_chunks_outside(db_session, corpus_celexes=[]) == 0
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


async def test_returns_only_the_chunks_without_a_vector(db_session, ingest_run, make_chunk_row):
    db_session.add(make_chunk_row(ingest_run, content_hash="a" * 64))
    db_session.add(make_chunk_row(ingest_run, content_hash="b" * 64, embedding=VECTOR))
    await db_session.flush()

    pending = await get_unembedded_chunks(db_session)

    assert [row.content_hash for row in pending] == ["a" * 64]


async def test_orders_by_document_then_insertion(db_session, ingest_run, make_chunk_row):
    """groupby in the stage only groups adjacent rows, so this ordering is load-bearing."""
    for celex, digest in [("32023R1805", "a"), ("32015R0757", "b"), ("32023R1805", "c")]:
        db_session.add(make_chunk_row(ingest_run, celex=celex, content_hash=digest * 64))
    await db_session.flush()

    pending = await get_unembedded_chunks(db_session)

    assert [row.celex for row in pending] == ["32015R0757", "32023R1805", "32023R1805"]
    assert [row.content_hash[0] for row in pending] == ["b", "a", "c"]


async def test_counts_only_the_chunks_that_carry_a_vector(db_session, ingest_run, make_chunk_row):
    db_session.add(make_chunk_row(ingest_run, content_hash="a" * 64))
    db_session.add(make_chunk_row(ingest_run, content_hash="b" * 64, embedding=VECTOR))
    db_session.add(make_chunk_row(ingest_run, content_hash="c" * 64, embedding=VECTOR))
    await db_session.flush()

    assert await count_embedded_chunks(db_session) == 2


async def test_counts_zero_on_an_empty_table(db_session):
    assert await count_embedded_chunks(db_session) == 0
