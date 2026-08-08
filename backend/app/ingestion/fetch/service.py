"""Queries over raw_documents: the corpus as it currently stands, and the fetch diff baseline."""

from collections.abc import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.ingestion.fetch.schemas import RawDocument


def latest_run_docs() -> Select[tuple[RawDocument]]:
    """Documents from each topic's own latest run: the corpus as it currently stands.

    The latest run is resolved per topic, so fetching topics separately still diffs.
    """
    other = aliased(RawDocument)
    latest_for_topic = (
        select(func.max(other.ingest_run_id))
        .where(other.topic == RawDocument.topic)
        .scalar_subquery()
    )
    return select(RawDocument).where(RawDocument.ingest_run_id == latest_for_topic)


async def get_corpus_docs(session: AsyncSession) -> Sequence[RawDocument]:
    """Every topic's latest documents, across all topics."""
    return (await session.scalars(latest_run_docs())).all()


async def get_other_topic_celexes(session: AsyncSession, topics: Sequence[str]) -> set[str]:
    """Celexes still held by the topics this run is not ingesting."""
    stmt = latest_run_docs().where(RawDocument.topic.notin_(topics))
    rows = await session.scalars(stmt)
    return {row.celex for row in rows}


async def get_baseline_docs(session: AsyncSession, topics: Sequence[str]) -> dict[str, RawDocument]:
    """Rows from each topic's own latest recorded run, keyed by celex."""
    stmt = (
        latest_run_docs().where(RawDocument.topic.in_(topics)).order_by(RawDocument.ingest_run_id)
    )
    rows = await session.scalars(stmt)
    return {row.celex: row for row in rows}
