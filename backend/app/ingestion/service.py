"""CRUD operations for the ingestion domain."""

import hashlib
import json
from collections.abc import Iterable, Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.clock import utc_now, utc_today
from app.core.db.crud import create_record, update_record
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.models import IngestRunUpdate
from app.ingestion.schemas import IngestRun


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
    return await update_ingest_run(
        session,
        run,
        IngestRunUpdate(status=status, completed_at=utc_now(), corpus_version=corpus_version),
    )


async def get_latest_corpus_version(session: AsyncSession) -> str | None:
    """The corpus version of the most recent run that was stamped with one."""
    return await session.scalar(
        select(IngestRun.corpus_version)
        .where(IngestRun.corpus_version.is_not(None))
        .order_by(IngestRun.id.desc())
        .limit(1)
    )


def corpus_fingerprint(documents: Iterable[RawDocument]) -> str:
    """Content hash of the corpus; an identical corpus fingerprints identically."""
    content = sorted((doc.ref, doc.resolved_ref, doc.sha256) for doc in documents)
    return hashlib.sha256(json.dumps(content).encode()).hexdigest()[:7]


async def next_corpus_version(session: AsyncSession) -> str:
    """Date the corpus last changed plus its fingerprint; unchanged corpora keep their version.

    Fingerprints the whole corpus, not just the topics fetched, so a single-topic run
    cannot mint a version that describes only part of it.
    """
    fingerprint = corpus_fingerprint(await get_corpus_docs(session))
    previous = await get_latest_corpus_version(session)
    if previous is not None and previous.endswith(f"-{fingerprint}"):
        return previous
    return f"{utc_today()}-{fingerprint}"


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


async def get_baseline_docs(session: AsyncSession, topics: Sequence[str]) -> dict[str, RawDocument]:
    """Rows from each topic's own latest recorded run, keyed by ref."""
    rows = await session.scalars(
        latest_run_docs().where(RawDocument.topic.in_(topics)).order_by(RawDocument.ingest_run_id)
    )
    return {row.ref: row for row in rows}
