"""Queries over raw_documents: the corpus as it stands, and what the previous run recorded."""

from sqlalchemy import Subquery, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import RawDocsQuery
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun


def _latest_success_run_per_topic() -> Subquery:
    """The highest successful run id for each topic, grouped once for the whole scan.

    Grouped rather than correlated on the outer row's topic: as a scalar subquery Postgres
    re-runs it for every row it scans, so the cost squares with the number of runs.
    """
    other = aliased(RawDocument)
    return (
        select(other.topic, func.max(other.ingest_run_id).label("run_id"))
        .join(IngestRun, IngestRun.id == other.ingest_run_id)
        .where(IngestRun.status == IngestRunStatus.SUCCESS)
        .group_by(other.topic)
        .subquery()
    )


async def get_raw_documents(session: AsyncSession, query: RawDocsQuery) -> dict[str, RawDocument]:
    """The newest standing row per celex, keyed by celex.

    Outer-joined so a topic with no successful run yet keeps every row it has, rather than
    dropping out of the corpus against a run id it cannot supply.
    """
    latest = _latest_success_run_per_topic()
    stmt = (
        select(RawDocument)
        .outerjoin(latest, latest.c.topic == RawDocument.topic)
        .distinct(RawDocument.celex)
        .where(RawDocument.ingest_run_id >= func.coalesce(latest.c.run_id, 0))
        .order_by(RawDocument.celex, RawDocument.ingest_run_id.desc())
    )
    if query.include_topics is not None:
        stmt = stmt.where(RawDocument.topic.in_(query.include_topics))
    if query.exclude_topics is not None:
        stmt = stmt.where(RawDocument.topic.notin_(query.exclude_topics))
    rows = await session.scalars(stmt)
    return {row.celex: row for row in rows}
