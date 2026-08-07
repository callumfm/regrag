"""The embed stage: how it batches, what it retries, and what a failure costs."""

from typing import Any

import pytest

from app.core.llm import LLMError
from app.ingestion.embed import stage as embed_stage
from app.ingestion.embed.service import get_unembedded_chunks
from app.ingestion.embed.stage import batches, embed_chunks

pytestmark = pytest.mark.anyio


class Row:
    """The only attribute `batches` reads, so the pure-batching tests need nothing else."""

    def __init__(self, celex: str):
        self.celex = celex


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch, defuse_retry):
    """Replace the provider with a scripted stub, recording the texts of each batch."""

    def _install(errors: dict[int, Exception] | None = None) -> list[list[str]]:
        calls: list[list[str]] = []
        failures = errors or {}

        async def fake_embed(texts: list[str], **kwargs: Any) -> list[list[float]]:
            calls.append(list(texts))
            if len(calls) in failures:
                raise failures[len(calls)]
            return [[float(index)] * 1024 for index in range(len(texts))]

        monkeypatch.setattr(embed_stage, "embed", fake_embed)
        defuse_retry(embed_stage.embed_texts)
        return calls

    return _install


def rows(make_chunk_row, run, celex: str, count: int, start: int = 0):
    """`count` chunk rows for one document, each with a distinct content hash."""
    return [
        make_chunk_row(run, celex=celex, content_hash=f"{celex}-{index + start}".ljust(64, "x"))
        for index in range(count)
    ]


def test_batches_split_at_the_provider_ceiling():
    produced = list(batches([Row("32023R1805")] * 129))  # ty: ignore[invalid-argument-type]

    assert [len(batch) for _, batch in produced] == [128, 1]


def test_batches_never_span_two_documents():
    rows_ = [Row("32015R0757")] * 3 + [Row("32023R1805")] * 2
    produced = list(batches(rows_))  # ty: ignore[invalid-argument-type]

    assert [(celex, len(batch)) for celex, batch in produced] == [
        ("32015R0757", 3),
        ("32023R1805", 2),
    ]


async def test_every_vectorless_chunk_gets_a_vector(
    db_session, ingest_run, make_chunk_row, provider
):
    provider()
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 3))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert (result.embedded, result.unchanged, result.failed) == (3, 0, {})
    assert await get_unembedded_chunks(db_session) == []


async def test_vectors_land_on_the_rows_in_input_order(
    db_session, ingest_run, make_chunk_row, provider
):
    """The stub numbers vectors 0..n within a batch, so a mis-zip shows up as a shuffle."""
    provider()
    created = rows(make_chunk_row, ingest_run, "32023R1805", 3)
    db_session.add_all(created)
    await db_session.flush()

    await embed_chunks(db_session)

    assert [row.embedding[0] for row in created] == [0.0, 1.0, 2.0]


async def test_already_embedded_chunks_are_reported_not_re_embedded(
    db_session, ingest_run, make_chunk_row, provider
):
    calls = provider()
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 2))
    db_session.add_all(
        [
            make_chunk_row(
                ingest_run, celex="32015R0757", content_hash="z" * 64, embedding=[0.5] * 1024
            )
        ]
    )
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert (result.embedded, result.unchanged) == (2, 1)
    assert len(calls) == 1


async def test_a_failed_batch_is_recorded_against_its_document(
    db_session, ingest_run, make_chunk_row, provider
):
    provider({1: LLMError("embedding call failed")})
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 2))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert result.embedded == 0
    assert list(result.failed) == ["32023R1805"]
    assert "LLMError" in result.failed["32023R1805"]


async def test_one_document_failing_does_not_stop_the_others(
    db_session, ingest_run, make_chunk_row, provider
):
    """Ordering is by celex, so 32015R0757's batch is call 1 and 32023R1805's is call 2."""
    provider({1: LLMError("embedding call failed")})
    db_session.add_all(rows(make_chunk_row, ingest_run, "32015R0757", 1))
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert list(result.failed) == ["32015R0757"]
    assert result.embedded == 1


async def test_a_transient_failure_is_retried(db_session, ingest_run, make_chunk_row, provider):
    calls = provider({1: LLMError("embedding call failed", transient=True)})
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert len(calls) == 2
    assert (result.embedded, result.failed) == (1, {})


async def test_a_permanent_failure_is_not_retried(db_session, ingest_run, make_chunk_row, provider):
    calls = provider({1: LLMError("embedding call failed")})
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 1))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert len(calls) == 1
    assert result.embedded == 0


async def test_a_later_batch_failing_keeps_the_earlier_batches_vectors(
    db_session, ingest_run, make_chunk_row, provider
):
    """The savepoint is per batch, so work already done survives a failure further in."""
    calls = provider({2: LLMError("embedding call failed")})
    db_session.add_all(rows(make_chunk_row, ingest_run, "32023R1805", 200))
    await db_session.flush()

    result = await embed_chunks(db_session)

    assert [len(batch) for batch in calls] == [128, 72]
    assert result.embedded == 128
    assert list(result.failed) == ["32023R1805"]
    assert len(await get_unembedded_chunks(db_session)) == 72


async def test_an_empty_table_makes_no_provider_call(db_session, provider):
    calls = provider()

    result = await embed_chunks(db_session)

    assert calls == []
    assert (result.embedded, result.unchanged) == (0, 0)
