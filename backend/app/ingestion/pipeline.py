"""The ingest pipeline: one run, discover -> fetch -> parse -> chunk -> embed."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import ObjectStore
from app.ingestion.chunk.stage import chunk_and_store_document, prune_chunks
from app.ingestion.discover.stage import discover_corpus
from app.ingestion.embed.stage import embed_chunks
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.service import get_other_topic_celexes, get_previous_docs
from app.ingestion.fetch.stage import fetch_document
from app.ingestion.parse.stage import parse_document
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
            session, run, status=IngestRunStatus.ABORTED, result=result.report()
        )
    except SQLAlchemyError:
        logger.exception("run %s could not be marked aborted", run_id)


async def celexes_to_keep(
    session: AsyncSession, *, result: IngestRunResult, topics: Sequence[str]
) -> set[str] | None:
    """The celexes the corpus consists of after this run: what it discovered, plus other topics'.

    None when any stage before chunk failed: pruning is irreversible, so a run with errors does
    not earn it.
    """
    if not (result.discover.ok and result.fetch.ok and result.parse.ok):
        return None
    return set(result.discover.celexes) | await get_other_topic_celexes(session, topics)


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
    store: ObjectStore,
) -> IngestRunResult:
    """Run the whole pipeline under one ingest run; blocking HTTP is fine here (CLI-only)."""
    async with ingest_run(session) as (run, result):
        previous = await get_previous_docs(session, topics)
        discovered, result.discover = discover_corpus(
            client, topics=topics, previous_celexes=previous
        )
        logger.info("[discover] %s", result.discover.summary())

        result.begin_document_stages()
        for document in discovered:
            fetched, fetch_result = await fetch_document(
                session,
                client=client,
                discovered=document,
                previous=previous.get(document.celex),
                run=run,
                store=store,
            )
            result.fetch += fetch_result
            if fetched is None:
                continue

            parsed, parse_result = parse_document(fetched)
            result.parse += parse_result
            if parsed is None:
                continue

            result.chunk += await chunk_and_store_document(session, parsed, ingest_run_id=run.id)
            await session.commit()

        logger.info("[fetch] %s", result.fetch.summary())
        logger.info("[parse] %s", result.parse.summary())

        corpus_celexes = await celexes_to_keep(session, result=result, topics=topics)
        if corpus_celexes is not None:
            result.chunk += await prune_chunks(session, corpus_celexes=corpus_celexes)
        logger.info("[chunk] %s", result.chunk.summary())

        result.embed = await embed_chunks(session)
        logger.info("[embed] %s", result.embed.summary())

    return result
