"""Queries over raw_documents: the corpus as it stands, and what the previous run recorded."""

from collections.abc import Sequence

from sqlalchemy import ColumnElement, ScalarSelect, Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.ingestion.enums import COMPLETE_CORPUS, IngestRunStatus
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun


def _latest_run_id(*predicates: ColumnElement[bool]) -> ScalarSelect[int]:
    """The highest run id holding rows for the outer row's own topic, or 0 if none qualifies."""
    other = aliased(RawDocument)
    stmt = (
        select(func.coalesce(func.max(other.ingest_run_id), 0))
        .join(IngestRun, IngestRun.id == other.ingest_run_id)
        .where(other.topic == RawDocument.topic, *predicates)
    )
    return stmt.scalar_subquery()


def latest_run_docs() -> Select[tuple[RawDocument]]:
    """Documents from each topic's own latest run, the one in flight included.

    Aborted runs are skipped: their rows are a prefix, so they would understate their topic.
    """
    latest = _latest_run_id(IngestRun.status != IngestRunStatus.ABORTED)
    return select(RawDocument).where(RawDocument.ingest_run_id == latest)


def standing_docs() -> Select[tuple[RawDocument]]:
    """Documents held from each topic's latest complete run on, incomplete runs included.

    A run that died mid-loop or failed a download still downloaded what it committed; only a
    run that got through its whole topic can retire the rows before it.
    """
    floor = _latest_run_id(IngestRun.status.in_(COMPLETE_CORPUS))
    return select(RawDocument).where(RawDocument.ingest_run_id >= floor)


async def get_corpus_docs(session: AsyncSession) -> Sequence[RawDocument]:
    """Every topic's latest documents, across all topics."""
    return (await session.scalars(latest_run_docs())).all()


async def get_other_topic_celexes(session: AsyncSession, topics: Sequence[str]) -> set[str]:
    """Celexes still held by the topics this run is not ingesting."""
    stmt = standing_docs().where(RawDocument.topic.notin_(topics))
    rows = await session.scalars(stmt)
    return {row.celex for row in rows}


async def get_previous_docs(session: AsyncSession, topics: Sequence[str]) -> dict[str, RawDocument]:
    """The newest row per celex these topics still hold, keyed by celex."""
    stmt = standing_docs().where(RawDocument.topic.in_(topics)).order_by(RawDocument.ingest_run_id)
    rows = await session.scalars(stmt)
    return {row.celex: row for row in rows}
