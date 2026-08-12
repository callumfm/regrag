"""The ingest pipeline: one run, discover -> fetch -> parse -> chunk -> embed."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import ObjectStore
from app.ingestion.chunk.service import delete_chunks_outside
from app.ingestion.chunk.stage import chunk_and_store_document
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.stage import discover_corpus
from app.ingestion.embed.stage import embed_chunks
from app.ingestion.enums import IngestRunStatus, Stage
from app.ingestion.exceptions import DocumentFailed
from app.ingestion.fetch.service import get_other_topic_celexes, get_previous_docs
from app.ingestion.fetch.stage import fetch_document
from app.ingestion.models import IngestRunResult
from app.ingestion.parse.stage import parse_document
from app.ingestion.schemas import IngestRun
from app.ingestion.service import complete_ingest_run, create_ingest_run

logger = logging.getLogger(__name__)


async def _mark_aborted(session: AsyncSession, run: IngestRun, result: IngestRunResult) -> None:
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
    session: AsyncSession, *, discovered: Sequence[DiscoveredDocument], topics: Sequence[str]
) -> set[str]:
    """The celexes the corpus consists of after this run: what it discovered, plus other topics'."""
    return {document.celex for document in discovered} | await get_other_topic_celexes(
        session, topics
    )


@asynccontextmanager
async def ingest_run(session: AsyncSession) -> AsyncIterator[tuple[IngestRun, IngestRunResult]]:
    """Open a run and close it out, marking it aborted if the body raises or is interrupted."""
    run = await create_ingest_run(session)
    result = IngestRunResult(run_id=run.id)
    try:
        yield run, result
        await complete_ingest_run(session, run, status=result.status, result=result.report())
    except BaseException:
        await _mark_aborted(session, run, result)
        raise
    result.corpus_version = run.corpus_version


async def _ingest_corpus(
    session: AsyncSession,
    run: IngestRun,
    result: IngestRunResult,
    *,
    client: httpx.AsyncClient,
    topics: Sequence[str],
    store: ObjectStore,
) -> None:
    """Fill in the run's result: discover, then a document at a time, then prune and embed."""
    previous = await get_previous_docs(session, topics)
    discovered, result.dropped = await discover_corpus(
        client, topics=topics, previous_celexes=previous
    )
    logger.info("%s", result.line(Stage.DISCOVER))

    for document in discovered:
        try:
            async with session.begin_nested():
                fetched, change = await fetch_document(
                    session,
                    client=client,
                    discovered=document,
                    previous=previous.get(document.celex),
                    run=run,
                    store=store,
                )
                parsed = parse_document(fetched)
                chunks = await chunk_and_store_document(session, parsed, ingest_run_id=run.id)
        except DocumentFailed as failure:
            result.fail(failure)
        else:
            await session.commit()
            result.record(document.celex, change=change, chunks=chunks)

    logger.info("%s", result.line(Stage.FETCH))
    logger.info("%s", result.line(Stage.PARSE))

    if result.corpus_complete:
        result.pruned = await delete_chunks_outside(
            session,
            corpus_celexes=await celexes_to_keep(session, discovered=discovered, topics=topics),
        )
    logger.info("%s", result.line(Stage.CHUNK))

    result.embed = await embed_chunks(session)
    logger.info("%s", result.line(Stage.EMBED))


async def ingest(
    session: AsyncSession,
    *,
    client: httpx.AsyncClient,
    topics: Sequence[str],
    store: ObjectStore,
) -> IngestRunResult:
    """Run the whole pipeline under one ingest run, each document landing whole or not at all."""
    async with ingest_run(session) as (run, result):
        await _ingest_corpus(session, run, result, client=client, topics=topics, store=store)
    return result
