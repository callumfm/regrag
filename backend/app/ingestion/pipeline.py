"""The ingest pipeline: one run, fetch -> parse -> chunk -> store."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.stage import chunk_documents
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.stage import fetch_documents
from app.ingestion.models import IngestRunResult
from app.ingestion.parse.stage import parse_documents
from app.ingestion.schemas import IngestRun
from app.ingestion.service import complete_ingest_run, create_ingest_run, next_corpus_version

logger = logging.getLogger(__name__)


@asynccontextmanager
async def ingest_run(session: AsyncSession) -> AsyncIterator[IngestRun]:
    """Open a run, and mark it failed if the pipeline raises out from under us."""
    run = await create_ingest_run(session)
    try:
        yield run
    except Exception:
        await complete_ingest_run(session, run, status=IngestRunStatus.FAILED)
        raise


async def ingest(
    session: AsyncSession,
    *,
    client: httpx.Client,
    topics: Sequence[str],
    data_dir: Path,
) -> IngestRunResult:
    """Run the whole pipeline under one ingest run; blocking HTTP is fine here (CLI-only)."""
    async with ingest_run(session) as run:
        documents, fetched = await fetch_documents(
            session, client=client, topics=topics, data_dir=data_dir, run=run
        )
        logger.info("[fetch] %s", fetched.summary())

        version = await next_corpus_version(session)

        parsed, parse_result = parse_documents(documents, data_dir=data_dir)
        logger.info("[parse] %s", parse_result.summary())

        chunked = await chunk_documents(
            session,
            parsed,
            corpus_version=version,
            topics=topics,
            discovered=fetched.discovered,
        )
        logger.info("[chunk] %s", chunked.summary())

        result = IngestRunResult(run_id=run.id, fetch=fetched, parse=parse_result, chunk=chunked)
        result.corpus_version = version if result.ok else None
        await complete_ingest_run(
            session, run, status=result.status, corpus_version=result.corpus_version
        )
        return result
