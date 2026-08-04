"""Roundtrip tests for the ingest tracking tables."""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.schemas import IngestedDocument, IngestRun

pytestmark = pytest.mark.anyio


def make_document(**overrides: Any) -> IngestedDocument:
    defaults: dict[str, Any] = {
        "name": "fueleu",
        "source": "eurlex",
        "ref": "32023R1805",
        "resolved_ref": "32023R1805",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023R1805",
        "sha256": "a" * 64,
        "size_bytes": 758462,
        "fetched_at": datetime.now(UTC),
        "topic": "fueleu",
    }
    return IngestedDocument(**{**defaults, **overrides})


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


async def test_ingested_document_belongs_to_run(db_session: AsyncSession):
    run = IngestRun(status=IngestRunStatus.COMPLETED, corpus_version="2026-08-03-abc1234")
    doc = make_document(run=run)
    db_session.add(doc)
    await db_session.flush()

    assert doc.ingest_run_id == run.id
    assert doc.run.corpus_version == "2026-08-03-abc1234"


async def test_document_name_unique_per_run(db_session: AsyncSession):
    run = IngestRun(status=IngestRunStatus.RUNNING)
    db_session.add(make_document(run=run))
    db_session.add(make_document(run=run))
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_document_carries_topic(db_session: AsyncSession):
    run = IngestRun(status=IngestRunStatus.RUNNING)
    doc = make_document(run=run, topic="fueleu")
    db_session.add(doc)
    await db_session.flush()

    assert doc.topic == "fueleu"
