"""CRUD operations for the ingestion domain."""

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

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
