"""CRUD operations for the ingestion domain."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

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
    session: AsyncSession, run: IngestRun, status: IngestRunStatus
) -> IngestRun:
    """Close out a run with its terminal status and completion timestamp."""
    return await update_ingest_run(
        session, run, IngestRunUpdate(status=status, completed_at=datetime.now(UTC))
    )


async def get_baseline_docs(
    session: AsyncSession, topics: Sequence[str]
) -> dict[str, IngestedDocument]:
    """Rows of the latest run that recorded documents, filtered to topics, keyed by name."""
    last_run_id = select(func.max(IngestedDocument.ingest_run_id)).scalar_subquery()
    rows = await session.scalars(
        select(IngestedDocument).where(
            IngestedDocument.ingest_run_id == last_run_id,
            IngestedDocument.topic.in_(topics),
        )
    )
    return {row.name: row for row in rows}
