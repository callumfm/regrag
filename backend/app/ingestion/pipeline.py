"""The ingest pipeline: one run, fetch -> parse -> chunk -> store."""

import logging
from collections.abc import Sequence
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import DiscoveryError
from app.ingestion.fetch.corpus import fetch_documents
from app.ingestion.models import RunReport
from app.ingestion.service import complete_ingest_run, create_ingest_run, next_corpus_version

logger = logging.getLogger(__name__)


async def ingest(
    session: AsyncSession,
    client: httpx.Client,
    topics: Sequence[str],
    data_dir: Path,
) -> RunReport:
    """Run the whole pipeline under one ingest run; blocking HTTP is fine here (CLI-only)."""
    run = await create_ingest_run(session)
    report = RunReport(run_id=run.id)
    try:
        await fetch_documents(session, client, topics, data_dir, run, report)
    except (DiscoveryError, httpx.HTTPError):
        await complete_ingest_run(session, run, IngestRunStatus.FAILED)
        raise
    logger.info(
        "[fetch] %d new, %d changed, %d unchanged, %d dropped, %d failed",
        len(report.new),
        len(report.changed),
        len(report.unchanged),
        len(report.dropped),
        len(report.failed),
    )
    report.corpus_version = await next_corpus_version(session) if report.ok else None
    await complete_ingest_run(session, run, report.status, report.corpus_version)
    return report
