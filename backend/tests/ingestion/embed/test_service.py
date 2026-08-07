"""Which chunks still need a vector, and how many already have one."""

import pytest

from app.ingestion.embed.service import count_embedded_chunks, get_unembedded_chunks

pytestmark = pytest.mark.anyio

VECTOR = [0.1] * 1024


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
