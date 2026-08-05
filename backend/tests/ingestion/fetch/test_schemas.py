"""Roundtrip tests for the raw documents table."""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.schemas import IngestRun

pytestmark = pytest.mark.anyio


async def test_document_belongs_to_run(db_session: AsyncSession, make_document):
    run = IngestRun(status=IngestRunStatus.COMPLETED, corpus_version="2026-08-03-abc1234")
    doc = make_document(run)
    db_session.add(doc)
    await db_session.flush()

    assert doc.ingest_run_id == run.id
    assert doc.run.corpus_version == "2026-08-03-abc1234"


async def test_document_ref_unique_per_run(db_session: AsyncSession, make_document):
    run = IngestRun(status=IngestRunStatus.RUNNING)
    db_session.add(make_document(run))
    db_session.add(make_document(run))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_document_carries_topic(db_session: AsyncSession, make_document):
    run = IngestRun(status=IngestRunStatus.RUNNING)
    doc = make_document(run, topic="fueleu")
    db_session.add(doc)
    await db_session.flush()

    assert doc.topic == "fueleu"
