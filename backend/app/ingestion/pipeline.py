"""The ingest pipeline: one run, fetch -> parse -> chunk -> embed."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.stage import chunk_and_store_documents
from app.ingestion.embed.stage import embed_chunks
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.fetch.service import get_other_topic_celexes
from app.ingestion.fetch.stage import fetch_documents
from app.ingestion.parse.models import ParseRunResult
from app.ingestion.parse.stage import parse_documents
from app.ingestion.result import IngestRunResult
from app.ingestion.schemas import IngestRun
from app.ingestion.service import complete_ingest_run, create_ingest_run

logger = logging.getLogger(__name__)


async def _mark_failed(session: AsyncSession, run: IngestRun, result: IngestRunResult) -> None:
    """Discard the aborted transaction, then close the run out without masking why it aborted."""
    run_id = run.id
    await session.rollback()
    try:
        await complete_ingest_run(
            session, run, status=IngestRunStatus.FAILED, result=result.report()
        )
    except SQLAlchemyError:
        logger.exception("run %s could not be marked failed", run_id)


async def _known_corpus_celexes(
    session: AsyncSession,
    *,
    fetch_result: FetchRunResult,
    parse_result: ParseRunResult,
    topics: Sequence[str],
) -> set[str] | None:
    """The celexes the corpus consists of after this run: what it discovered, plus other topics'.

    None when any stage failed: pruning is irreversible, so a run with errors does not earn it.
    """
    if not (fetch_result.ok and parse_result.ok):
        return None
    return set(fetch_result.discovered) | await get_other_topic_celexes(session, topics)


@asynccontextmanager
async def ingest_run(session: AsyncSession) -> AsyncIterator[tuple[IngestRun, IngestRunResult]]:
    """Open a run and close it out, marking it failed if the body raises or is interrupted."""
    run = await create_ingest_run(session)
    result = IngestRunResult(run_id=run.id)
    try:
        yield run, result
        await complete_ingest_run(session, run, status=result.status, result=result.report())
    except BaseException:
        await _mark_failed(session, run, result)
        raise
    result.corpus_version = run.corpus_version


async def ingest(
    session: AsyncSession,
    *,
    client: httpx.Client,
    topics: Sequence[str],
    data_dir: Path,
) -> IngestRunResult:
    """Run the whole pipeline under one ingest run; blocking HTTP is fine here (CLI-only)."""
    async with ingest_run(session) as (run, result):
        documents, result.fetch = await fetch_documents(
            session, client=client, topics=topics, data_dir=data_dir, run=run
        )
        logger.info("[fetch] %s", result.fetch.summary())

        parsed, result.parse = parse_documents(documents, data_dir=data_dir)
        logger.info("[parse] %s", result.parse.summary())

        corpus_celexes = await _known_corpus_celexes(
            session, fetch_result=result.fetch, parse_result=result.parse, topics=topics
        )
        result.chunk = await chunk_and_store_documents(
            session, parsed, ingest_run_id=run.id, corpus_celexes=corpus_celexes
        )
        logger.info("[chunk] %s", result.chunk.summary())

        result.embed = await embed_chunks(session)
        logger.info("[embed] %s", result.embed.summary())

    return result
