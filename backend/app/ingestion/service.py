"""CRUD operations for the ingestion domain."""

import hashlib
import json
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now, utc_today
from app.core.db.crud import create_record, update_record
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_corpus_docs
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


async def complete_ingest_run(
    session: AsyncSession, run: IngestRun, *, status: IngestRunStatus
) -> IngestRun:
    """Close out a run, minting a corpus version for it only if every stage succeeded."""
    version = await next_corpus_version(session) if status is IngestRunStatus.COMPLETED else None
    return await update_ingest_run(
        session,
        run,
        IngestRunUpdate(status=status, completed_at=utc_now(), corpus_version=version),
    )
