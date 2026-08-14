"""Queries over raw_documents: the corpus as it stands, and what the previous run recorded."""

from sqlalchemy import ScalarSelect, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import RawDocsQuery
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun


def _latest_success_run_id() -> ScalarSelect[int]:
    """The highest successful run id holding rows for the outer row's own topic, or 0 if none."""
    other = aliased(RawDocument)
    stmt = (
        select(func.coalesce(func.max(other.ingest_run_id), 0))
        .join(IngestRun, IngestRun.id == other.ingest_run_id)
        .where(other.topic == RawDocument.topic, IngestRun.status == IngestRunStatus.SUCCESS)
    )
    return stmt.scalar_subquery()


async def get_raw_documents(session: AsyncSession, query: RawDocsQuery) -> dict[str, RawDocument]:
    """The newest standing row per celex, keyed by celex."""
    stmt = (
        select(RawDocument)
        .distinct(RawDocument.celex)
        .where(RawDocument.ingest_run_id >= _latest_success_run_id())
        .order_by(RawDocument.celex, RawDocument.ingest_run_id.desc())
    )
    if query.include_topics is not None:
        stmt = stmt.where(RawDocument.topic.in_(query.include_topics))
    if query.exclude_topics is not None:
        stmt = stmt.where(RawDocument.topic.notin_(query.exclude_topics))
    rows = await session.scalars(stmt)
    return {row.celex: row for row in rows}
