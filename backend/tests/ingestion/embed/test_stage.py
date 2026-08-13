"""The embed stage: how it batches, what it retries, and what a failure costs."""

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import EMBED_DIMENSIONS, config
from app.core.llm import LLMError
from app.ingestion.chunk.models import ChunkQuery
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.service import get_chunks
from app.ingestion.embed import stage
from app.ingestion.embed.stage import ChunkToEmbed, _batch_by_document, embed_chunks

pytestmark = pytest.mark.anyio


def rows(make_chunk_row, run, celex: str, count: int, start: int = 0):
    """`count` chunk rows for one document, each with a distinct content hash."""
    return [
        make_chunk_row(run, celex=celex, content_hash=f"{celex}-{index + start}".ljust(64, "x"))
        for index in range(count)
    ]


def test_batches_split_at_the_provider_ceiling():
    produced = list(_batch_by_document([ChunkToEmbed(id=0, celex="32023R1805", text="")] * 129))

    assert [len(batch) for _, batch in produced] == [128, 1]


def test_batches_never_span_two_documents():
    chunks = [ChunkToEmbed(id=0, celex="32015R0757", text="")] * 3 + [
        ChunkToEmbed(id=0, celex="32023R1805", text="")
    ] * 2
    produced = list(_batch_by_document(chunks))

    assert [(celex, len(batch)) for celex, batch in produced] == [
        ("32015R0757", 3),
        ("32023R1805", 2),
    ]


async def test_every_vectorless_chunk_gets_a_vector(db_session, ingest_run, make_chunk_row):
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 3))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert (result.embedded, result.already_embedded, result.failed) == (3, 0, {})
    assert await get_chunks(db_session, ChunkQuery(has_embedding=False, limit=100)) == []


async def test_vectors_land_on_the_rows_in_input_order(db_session, ingest_run, make_chunk_row):
    """The stub numbers vectors 0..n within a batch, so a mis-zip shows up as a shuffle."""
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 3))
    await db_session.flush()

    await embed_chunks(db_session)

    stored = await db_session.scalars(select(DocumentChunk.embedding).order_by(DocumentChunk.id))
    assert [vector[0] for vector in stored.all()] == [0.0, 1.0, 2.0]


async def test_already_embedded_chunks_are_reported_not_re_embedded(
    db_session, ingest_run, make_chunk_row, embeddings
):
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 2))
    db_session.add(
        make_chunk_row(
            ingest_run,
            celex="32015R0757",
            content_hash="z" * 64,
            embedding=[0.5] * EMBED_DIMENSIONS,
        )
    )
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert (result.embedded, result.already_embedded) == (2, 1)
    assert len(embeddings.calls) == 1


async def test_a_failed_batch_is_recorded_against_its_document(
    db_session, ingest_run, make_chunk_row, embeddings
):
    embeddings.errors[1] = LLMError("embedding call failed")
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 2))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert result.embedded == 0
    assert result.failures == {"32023R1805": "2 chunks: LLMError: embedding call failed"}


async def test_every_failed_batch_of_a_document_counts_towards_its_loss(
    db_session, ingest_run, make_chunk_row, embeddings
):
    """A document spanning several batches reports every chunk it lost, not just the last one's."""
    embeddings.errors[1] = LLMError("embedding call failed")
    embeddings.errors[2] = LLMError("embedding call failed")
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 200))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert result.embedded == 0
    assert result.failures == {"32023R1805": "200 chunks: LLMError: embedding call failed"}


async def test_one_document_failing_does_not_stop_the_others(
    db_session, ingest_run, make_chunk_row, embeddings
):
    """Ordering is by celex, so 32015R0757's batch is call 1 and 32023R1805's is call 2."""
    embeddings.errors[1] = LLMError("embedding call failed")
    db_session.add_all(rows(make_chunk_row, ingest_run, "32015R0757", 1))
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert list(result.failed) == ["32015R0757"]
    assert result.embedded == 1


async def test_a_transient_failure_is_retried(db_session, ingest_run, make_chunk_row, embeddings):
    embeddings.errors[1] = LLMError("embedding call failed", transient=True)
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert len(embeddings.calls) == 2
    assert (result.embedded, result.failed) == (1, {})


async def test_a_permanent_failure_is_not_retried(
    db_session, ingest_run, make_chunk_row, embeddings
):
    embeddings.errors[1] = LLMError("embedding call failed")
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert len(embeddings.calls) == 1
    assert result.embedded == 0


async def test_a_later_batch_failing_keeps_the_earlier_batches_vectors(
    db_session, ingest_run, make_chunk_row, embeddings
):
    """Each batch commits alone, so work already done survives a failure further in."""
    embeddings.errors[2] = LLMError("embedding call failed")
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 200))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert [len(batch) for batch in embeddings.calls] == [128, 72]
    assert result.embedded == 128
    assert list(result.failed) == ["32023R1805"]
    assert len(await get_chunks(db_session, ChunkQuery(has_embedding=False, limit=100))) == 72


async def test_an_empty_table_makes_no_provider_call(db_session, embeddings):
    result = await embed_chunks(db_session)

    assert embeddings.calls == []
    assert (result.embedded, result.already_embedded) == (0, 0)


async def test_a_corpus_larger_than_one_page_ends_with_every_chunk_embedded(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    """The sweep writes vectors as it reads, so the cursor must not skip rows beneath it."""
    monkeypatch.setattr(config, "EMBED_PAGE_SIZE", 2)
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 7))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert result.embedded == 7
    assert await get_chunks(db_session, ChunkQuery(has_embedding=False, limit=100)) == []


async def test_a_batch_that_keeps_failing_does_not_loop_the_sweep(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    """A failed chunk stays vectorless, so a cursor that did not advance would never finish."""
    sizes: list[int] = []

    async def always_fails(texts, input_type):
        sizes.append(len(texts))
        raise LLMError("provider down")

    monkeypatch.setattr("app.ingestion.embed.stage.embed", always_fails)
    monkeypatch.setattr(config, "EMBED_PAGE_SIZE", 2)
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 5))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert sizes == [2, 2, 1]
    assert result.embedded == 0
    assert list(result.failed) == ["32023R1805"]


async def test_a_database_failure_in_one_batch_does_not_kill_the_sweep(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    """A failed write rolls back alone: the cursor was read before it, later batches commit."""
    monkeypatch.setattr(config, "EMBED_PAGE_SIZE", 2)
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 4))
    await db_session.flush()

    calls: list[int] = []
    real_write = stage.update_chunks

    async def fail_once(session, updates):
        calls.append(1)
        if len(calls) == 1:
            raise SQLAlchemyError("connection lost")
        await real_write(session, updates)

    monkeypatch.setattr("app.ingestion.embed.stage.update_chunks", fail_once)

    result = await embed_chunks(db_session)

    assert list(result.failed) == ["32023R1805"]
    assert result.embedded == 2


def overlap_tracker(peaks: list[int]):
    """An embed stub that records how many calls are in flight when each call runs."""
    active = 0

    async def tracking(texts, *, input_type):
        nonlocal active
        active += 1
        await asyncio.sleep(0)
        peaks.append(active)
        active -= 1
        return [[0.0] * EMBED_DIMENSIONS] * len(texts)

    return tracking


async def test_provider_calls_overlap_within_a_page(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    peaks: list[int] = []
    monkeypatch.setattr("app.ingestion.embed.stage.embed", overlap_tracker(peaks))
    db_session.add_all(rows(make_chunk_row, ingest_run, "32015R0757", 1))
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert max(peaks) == 2
    assert (result.embedded, result.failed) == (2, {})


async def test_in_flight_provider_calls_are_capped(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    peaks: list[int] = []
    monkeypatch.setattr("app.ingestion.embed.stage.embed", overlap_tracker(peaks))
    monkeypatch.setattr(config, "EMBED_CONCURRENCY", 2)
    for celex in ["32015R0757", "32023R1805", "32024R0001"]:
        db_session.add_all(rows(make_chunk_row, ingest_run, celex, 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert max(peaks) == 2
    assert result.embedded == 3
