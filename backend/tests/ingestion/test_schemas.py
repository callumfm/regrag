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
    assert fetched.created_at is not None
