"""The embed stage: how it batches, what it retries, and what a failure costs."""

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import EMBED_DIMENSIONS
from app.core.llm import LLMError
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.chunk.service import get_unembedded_chunks
from app.ingestion.embed.stage import _batches, embed_chunks

pytestmark = pytest.mark.anyio


def rows(make_chunk_row, run, celex: str, count: int, start: int = 0):
    """`count` chunk rows for one document, each with a distinct content hash."""
    return [
        make_chunk_row(run, celex=celex, content_hash=f"{celex}-{index + start}".ljust(64, "x"))
        for index in range(count)
    ]


def test_batches_split_at_the_provider_ceiling():
    produced = list(_batches([DocumentChunk(celex="32023R1805")] * 129))

    assert [len(batch) for _, batch in produced] == [128, 1]


def test_batches_never_span_two_documents():
    chunks = [DocumentChunk(celex="32015R0757")] * 3 + [DocumentChunk(celex="32023R1805")] * 2
    produced = list(_batches(chunks))

    assert [(celex, len(batch)) for celex, batch in produced] == [
        ("32015R0757", 3),
        ("32023R1805", 2),
    ]


async def test_every_vectorless_chunk_gets_a_vector(db_session, ingest_run, make_chunk_row):
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 3))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert (result.embedded, result.unchanged, result.failed) == (3, 0, {})
    assert await get_unembedded_chunks(db_session, limit=100) == []


async def test_vectors_land_on_the_rows_in_input_order(db_session, ingest_run, make_chunk_row):
    """The stub numbers vectors 0..n within a batch, so a mis-zip shows up as a shuffle."""
    created = rows(make_chunk_row, ingest_run, "32023R1805", 3)
    db_session.add_all(created)
    await db_session.flush()

    await embed_chunks(db_session)

    assert [row.embedding[0] for row in created] == [0.0, 1.0, 2.0]


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

    assert (result.embedded, result.unchanged) == (2, 1)
    assert len(embeddings.calls) == 1


async def test_a_failed_batch_is_recorded_against_its_document(
    db_session, ingest_run, make_chunk_row, embeddings
):
    embeddings.errors[1] = LLMError("embedding call failed")
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 2))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert result.embedded == 0
    assert list(result.failed) == ["32023R1805"]
    assert "LLMError" in result.failed["32023R1805"]


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
    """The savepoint is per batch, so work already done survives a failure further in."""
    embeddings.errors[2] = LLMError("embedding call failed")
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 200))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert [len(batch) for batch in embeddings.calls] == [128, 72]
    assert result.embedded == 128
    assert list(result.failed) == ["32023R1805"]
    assert len(await get_unembedded_chunks(db_session, limit=100)) == 72


async def test_an_empty_table_makes_no_provider_call(db_session, embeddings):
    result = await embed_chunks(db_session)

    assert embeddings.calls == []
    assert (result.embedded, result.unchanged) == (0, 0)


async def test_a_corpus_larger_than_one_page_ends_with_every_chunk_embedded(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    """The sweep writes vectors as it reads, so the cursor must not skip rows beneath it."""
    monkeypatch.setattr("app.ingestion.embed.stage.EMBED_PAGE_SIZE", 2)
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 7))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert result.embedded == 7
    assert await get_unembedded_chunks(db_session, limit=100) == []


async def test_a_batch_that_keeps_failing_does_not_loop_the_sweep(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    """A failed chunk stays vectorless, so a cursor that did not advance would never finish."""
    sizes: list[int] = []

    async def always_fails(texts, input_type):
        sizes.append(len(texts))
        raise LLMError("provider down")

    monkeypatch.setattr("app.ingestion.embed.stage.embed", always_fails)
    monkeypatch.setattr("app.ingestion.embed.stage.EMBED_PAGE_SIZE", 2)
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 5))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert sizes == [2, 2, 1]
    assert result.embedded == 0
    assert list(result.failed) == ["32023R1805"]


async def test_a_database_failure_in_the_last_batch_does_not_kill_the_sweep(
    db_session, ingest_run, make_chunk_row, monkeypatch
):
    """A savepoint rollback expires the page's rows, so the cursor must be read before the yield."""
    monkeypatch.setattr("app.ingestion.embed.stage.EMBED_PAGE_SIZE", 2)
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 4))
    await db_session.flush()

    calls: list[int] = []
    real_flush = db_session.flush

    async def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise SQLAlchemyError("connection lost")
        return await real_flush(*args, **kwargs)

    monkeypatch.setattr(db_session, "flush", fail_once)

    result = await embed_chunks(db_session)

    assert list(result.failed) == ["32023R1805"]
    assert result.embedded == 2
