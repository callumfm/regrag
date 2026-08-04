"""CRUD operations for the ingestion domain."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.crud import create_record, update_record
from app.db.schemas import IngestedDocument, IngestRun
from app.ingestion.enums import IngestRunStatus


async def create_ingest_run(session: AsyncSession) -> IngestRun:
    """Open a run in the RUNNING state and flush it so it has an id."""
    return await create_record(session, IngestRun(status=IngestRunStatus.RUNNING))


async def update_ingest_run(
    session: AsyncSession,
    run: IngestRun,
    *,
    status: IngestRunStatus | None = None,
    corpus_version: str | None = None,
    completed_at: datetime | None = None,
) -> IngestRun:
    """Patch a run's mutable fields; arguments left as None are untouched."""
    updates = {"status": status, "corpus_version": corpus_version, "completed_at": completed_at}
    return await update_record(session, run, {k: v for k, v in updates.items() if v is not None})


async def complete_ingest_run(
    session: AsyncSession, run: IngestRun, status: IngestRunStatus
) -> IngestRun:
    """Close out a run with its terminal status and completion timestamp."""
    return await update_ingest_run(session, run, status=status, completed_at=datetime.now(UTC))


async def get_baseline_docs(
    session: AsyncSession, topics: Sequence[str]
) -> dict[str, IngestedDocument]:
    """Rows of the latest run that recorded documents, filtered to topics, keyed by name."""
    last_run_id = await session.scalar(select(func.max(IngestedDocument.ingest_run_id)))
    if last_run_id is None:
        return {}
    rows = await session.scalars(
        select(IngestedDocument).where(
            IngestedDocument.ingest_run_id == last_run_id,
            IngestedDocument.topic.in_(topics),
        )
    )
    return {row.name: row for row in rows}
