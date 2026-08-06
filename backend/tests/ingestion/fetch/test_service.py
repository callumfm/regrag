"""The fetch diff baseline query."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.schemas import IngestRun

pytestmark = pytest.mark.anyio


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


async def test_baseline_survives_a_newer_run_of_another_topic(
    db_session: AsyncSession, make_document
):
    mrv_run = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_document(mrv_run, "32015R0757", topic="mrv"))
    await db_session.flush()

    fueleu_run = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_document(fueleu_run, "32023R1805", topic="fueleu"))
    await db_session.flush()

    assert fueleu_run.id > mrv_run.id
    assert set(await get_baseline_docs(db_session, ["mrv"])) == {"32015R0757"}


async def test_baseline_skips_newer_run_without_rows(db_session: AsyncSession, make_document):
    with_rows = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_document(with_rows, "32015R0757", topic="mrv"))
    db_session.add(IngestRun(status=IngestRunStatus.FAILED))
    await db_session.flush()

    assert set(await get_baseline_docs(db_session, ["mrv"])) == {"32015R0757"}
