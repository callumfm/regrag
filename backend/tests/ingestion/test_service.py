"""Ingestion CRUD: run lifecycle, the corpus fingerprint and its version."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_today
from app.ingestion.enums import IngestRunStatus
from app.ingestion.schemas import IngestRun
from app.ingestion.service import (
    complete_ingest_run,
    corpus_fingerprint,
    create_ingest_run,
    next_corpus_version,
)

pytestmark = pytest.mark.anyio


async def test_create_run_starts_running_with_an_id(db_session: AsyncSession):
    run = await create_ingest_run(db_session)
    assert run.id is not None
    assert run.status is IngestRunStatus.RUNNING
    assert run.completed_at is None


async def test_complete_run_sets_status_and_timestamp(db_session: AsyncSession):
    run = await create_ingest_run(db_session)
    await complete_ingest_run(db_session, run, status=IngestRunStatus.FAILED)
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None


async def test_complete_run_stores_the_stage_report_it_is_given(db_session: AsyncSession):
    run = await create_ingest_run(db_session)
    report = {"fetch": {"new": 1, "failed": {}}, "parse": {"parsed": 1, "failed": {}}}

    await complete_ingest_run(db_session, run, status=IngestRunStatus.SUCCESS, result=report)

    assert run.result == report


async def test_a_run_closed_without_a_report_records_none(db_session: AsyncSession):
    """An aborted run stores NULL, which means 'died before a result existed'."""
    run = await create_ingest_run(db_session)

    await complete_ingest_run(db_session, run, status=IngestRunStatus.FAILED)

    assert run.result is None


async def test_a_failed_run_is_not_given_a_corpus_version(db_session: AsyncSession, make_document):
    run = await create_ingest_run(db_session)
    db_session.add(make_document(run, "32015R0757"))
    await db_session.flush()

    await complete_ingest_run(db_session, run, status=IngestRunStatus.FAILED)

    assert run.corpus_version is None


async def test_a_completed_run_mints_a_corpus_version(db_session: AsyncSession, make_document):
    run = await create_ingest_run(db_session)
    db_session.add(make_document(run, "32015R0757"))
    await db_session.flush()

    await complete_ingest_run(db_session, run, status=IngestRunStatus.SUCCESS)

    assert run.corpus_version == await next_corpus_version(db_session)


async def test_fingerprint_ignores_document_order(db_session: AsyncSession, make_document):
    run = IngestRun(status=IngestRunStatus.SUCCESS)
    docs = [make_document(run, "32015R0757"), make_document(run, "32023R1805")]
    assert corpus_fingerprint(docs) == corpus_fingerprint(list(reversed(docs)))


async def test_fingerprint_changes_when_a_document_content_hash_changes(
    db_session: AsyncSession, make_document
):
    run = IngestRun(status=IngestRunStatus.SUCCESS)
    before = [make_document(run, "32015R0757")]
    after = [make_document(run, "32015R0757", sha256="b" * 64)]
    assert corpus_fingerprint(before) != corpus_fingerprint(after)


async def test_first_version_is_dated_today(db_session: AsyncSession, make_document):
    run = IngestRun(status=IngestRunStatus.SUCCESS)
    doc = make_document(run, "32015R0757")
    db_session.add(doc)
    await db_session.flush()

    assert await next_corpus_version(db_session) == f"{utc_today()}-{corpus_fingerprint([doc])}"


async def test_unchanged_corpus_keeps_the_previous_version(db_session: AsyncSession, make_document):
    run = IngestRun(status=IngestRunStatus.SUCCESS)
    doc = make_document(run, "32015R0757")
    stamped = IngestRun(
        status=IngestRunStatus.SUCCESS,
        corpus_version=f"2020-01-01-{corpus_fingerprint([doc])}",
    )
    db_session.add_all([doc, stamped])
    await db_session.flush()

    assert await next_corpus_version(db_session) == stamped.corpus_version


async def test_changed_corpus_gets_a_freshly_dated_version(db_session: AsyncSession, make_document):
    run = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add_all(
        [
            make_document(run, "32015R0757"),
            IngestRun(status=IngestRunStatus.SUCCESS, corpus_version="2020-01-01-0000000"),
        ]
    )
    await db_session.flush()

    assert (await next_corpus_version(db_session)).startswith(f"{utc_today()}-")


async def test_version_covers_topics_the_run_did_not_fetch(db_session: AsyncSession, make_document):
    """Re-fetching one topic must not mint a version describing only that topic."""
    first = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add_all(
        [
            make_document(first, "32015R0757", topic="mrv"),
            make_document(first, "32023R1805", topic="fueleu"),
        ]
    )
    await db_session.flush()
    whole_corpus = await next_corpus_version(db_session)

    second = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add(make_document(second, "32015R0757", topic="mrv"))
    await db_session.flush()

    assert await next_corpus_version(db_session) == whole_corpus
