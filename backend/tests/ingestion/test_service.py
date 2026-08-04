"""Ingestion CRUD: run lifecycle and the baseline document query."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.models import IngestRunUpdate
from app.ingestion.schemas import IngestRun
from app.ingestion.service import (
    complete_ingest_run,
    create_ingest_run,
    get_baseline_docs,
    update_ingest_run,
)

pytestmark = pytest.mark.anyio


async def test_create_run_starts_running_with_an_id(db_session: AsyncSession):
    run = await create_ingest_run(db_session)
    assert run.id is not None
    assert run.status is IngestRunStatus.RUNNING
    assert run.completed_at is None


async def test_update_run_leaves_omitted_fields_untouched(db_session: AsyncSession):
    run = await create_ingest_run(db_session)
    await update_ingest_run(db_session, run, IngestRunUpdate(corpus_version="2026-08-04-abc1234"))
    assert run.corpus_version == "2026-08-04-abc1234"
    assert run.status is IngestRunStatus.RUNNING


async def test_complete_run_sets_status_and_timestamp(db_session: AsyncSession):
    run = await create_ingest_run(db_session)
    await complete_ingest_run(db_session, run, IngestRunStatus.FAILED)
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None


async def test_baseline_empty_when_no_prior_runs(db_session: AsyncSession):
    assert await get_baseline_docs(db_session, ["mrv"]) == {}


async def test_baseline_is_latest_run_with_rows_filtered_to_topics(
    db_session: AsyncSession, make_document
):
    old = IngestRun(status=IngestRunStatus.COMPLETED)
    latest = IngestRun(status=IngestRunStatus.FAILED)
    db_session.add_all(
        [
            make_document(old, "32014R0666", topic="mrv"),
            make_document(latest, "32015R0757", topic="mrv", resolved_ref="02015R0757-20250101"),
            make_document(latest, "32023R1805", topic="fueleu"),
        ]
    )
    await db_session.flush()

    baseline = await get_baseline_docs(db_session, ["mrv"])
    assert set(baseline) == {"32015R0757"}
    assert baseline["32015R0757"].resolved_ref == "02015R0757-20250101"


async def test_baseline_skips_newer_run_without_rows(db_session: AsyncSession, make_document):
    with_rows = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_document(with_rows, "32015R0757", topic="mrv"))
    db_session.add(IngestRun(status=IngestRunStatus.FAILED))
    await db_session.flush()

    assert set(await get_baseline_docs(db_session, ["mrv"])) == {"32015R0757"}
