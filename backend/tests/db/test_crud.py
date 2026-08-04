"""Generic persistence helpers."""

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import create_record, update_record
from app.db.schemas import IngestRun
from app.ingestion.enums import IngestRunStatus

pytestmark = pytest.mark.anyio


async def test_create_record_populates_server_defaults(db_session: AsyncSession):
    run = await create_record(db_session, IngestRun(status=IngestRunStatus.RUNNING))
    assert run.id is not None
    assert run.created_at is not None


async def test_create_record_without_commit_still_assigns_id(db_session: AsyncSession):
    run = await create_record(db_session, IngestRun(status=IngestRunStatus.RUNNING), commit=False)
    assert run.id is not None


async def test_update_record_applies_only_given_fields(db_session: AsyncSession):
    run = await create_record(db_session, IngestRun(status=IngestRunStatus.RUNNING))
    await update_record(db_session, run, {"corpus_version": "2026-08-04-abc1234"})
    assert run.corpus_version == "2026-08-04-abc1234"
    assert run.status is IngestRunStatus.RUNNING


async def test_update_record_with_empty_updates_is_a_no_op(db_session: AsyncSession):
    run = await create_record(db_session, IngestRun(status=IngestRunStatus.RUNNING))
    await update_record(db_session, run, {})
    assert run.status is IngestRunStatus.RUNNING


class CorpusVersionUpdate(BaseModel):
    """Throwaway update model exercising update_record's Pydantic path."""

    corpus_version: str | None = None


async def test_update_record_from_model_ignores_omitted_fields(db_session: AsyncSession):
    run = IngestRun(status=IngestRunStatus.RUNNING, corpus_version="v1")
    await create_record(db_session, run)
    await update_record(db_session, run, CorpusVersionUpdate())
    assert run.corpus_version == "v1"


async def test_update_record_from_model_nulls_explicitly_passed_none(db_session: AsyncSession):
    run = IngestRun(status=IngestRunStatus.RUNNING, corpus_version="v1")
    await create_record(db_session, run)
    await update_record(db_session, run, CorpusVersionUpdate(corpus_version=None))
    assert run.corpus_version is None
