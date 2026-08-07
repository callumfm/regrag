"""Roundtrip tests for the ingest run tracking table."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.schemas import IngestRun

pytestmark = pytest.mark.anyio


async def test_ingest_run_roundtrip(db_session: AsyncSession):
    run = IngestRun(status=IngestRunStatus.RUNNING)
    db_session.add(run)
    await db_session.flush()
    run_id = run.id
    db_session.expire_all()

    fetched = (await db_session.scalars(select(IngestRun))).one()
    assert fetched.id == run_id
    assert fetched.status is IngestRunStatus.RUNNING
    assert fetched.corpus_version is None
    assert fetched.completed_at is None
    assert fetched.result is None
    assert fetched.created_at is not None


async def test_ingest_run_result_roundtrips_as_json(db_session: AsyncSession):
    report = {
        "fetch": {"new": 1, "changed": 0, "unchanged": 4, "dropped": 0, "failed": {}},
        "parse": {"parsed": 4, "failed": {"32023R2917": "ParseError: unrecognised dialect"}},
    }
    db_session.add(IngestRun(status=IngestRunStatus.FAILED, result=report))
    await db_session.flush()
    db_session.expire_all()

    assert (await db_session.scalars(select(IngestRun))).one().result == report
