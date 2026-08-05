"""CRUD operations for the ingestion domain."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.clock import utc_now
from app.core.db.crud import create_record, update_record
from app.ingestion.enums import IngestRunStatus
from app.ingestion.models import IngestRunUpdate
from app.ingestion.schemas import IngestedDocument, IngestRun


async def create_ingest_run(session: AsyncSession) -> IngestRun:
    """Open a run in the RUNNING state, committed so it outlives the fetch that follows."""
    return await create_record(session, IngestRun(status=IngestRunStatus.RUNNING))


async def update_ingest_run(
    session: AsyncSession, run: IngestRun, update_in: IngestRunUpdate
) -> IngestRun:
    """Partially update an ingest run."""
    return await update_record(session, run, update_in)


async def complete_ingest_run(
    session: AsyncSession,
    run: IngestRun,
    status: IngestRunStatus,
    corpus_version: str | None = None,
) -> IngestRun:
    """Close out a run with its terminal status, completion time and corpus version."""
    fields: dict[str, Any] = {"status": status, "completed_at": utc_now()}
    if corpus_version is not None:
        fields["corpus_version"] = corpus_version
    return await update_ingest_run(session, run, IngestRunUpdate(**fields))


async def get_latest_corpus_version(session: AsyncSession) -> str | None:
    """The corpus version of the most recent run that was stamped with one."""
    return await session.scalar(
        select(IngestRun.corpus_version)
        .where(IngestRun.corpus_version.is_not(None))
        .order_by(IngestRun.id.desc())
        .limit(1)
    )


async def get_baseline_docs(
    session: AsyncSession, topics: Sequence[str]
) -> dict[str, IngestedDocument]:
    """Rows from each topic's own latest recorded run, keyed by name.

    The latest run is resolved per topic, so fetching topics separately still diffs.
    """
    other = aliased(IngestedDocument)
    latest_for_topic = (
        select(func.max(other.ingest_run_id))
        .where(other.topic == IngestedDocument.topic)
        .scalar_subquery()
    )
    rows = await session.scalars(
        select(IngestedDocument)
        .where(
            IngestedDocument.topic.in_(topics),
            IngestedDocument.ingest_run_id == latest_for_topic,
        )
        .order_by(IngestedDocument.ingest_run_id)
    )
    return {row.name: row for row in rows}
