"""The ingest pipeline: one run, discover -> fetch -> parse -> chunk -> embed."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import ObjectStore
from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.chunk.stage import chunk_and_store_document, prune_chunks
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.stage import discover_corpus
from app.ingestion.embed.stage import embed_chunks
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_other_topic_celexes, get_previous_docs
from app.ingestion.fetch.stage import fetch_document
from app.ingestion.parse.models import ParseRunResult
from app.ingestion.parse.stage import parse_document
from app.ingestion.result import IngestRunResult
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
    session: AsyncSession,
    *,
    discovered: Sequence[DiscoveredDocument],
    result: IngestRunResult,
    topics: Sequence[str],
) -> set[str] | None:
    """The celexes the corpus consists of after this run: what it discovered, plus other topics'.

    None when fetch or parse failed: pruning is irreversible, and discovery either enumerates
    the corpus or raises, so there is no partial discovery to guard against.
    """
    if not (result.fetch.ok and result.parse.ok):
        return None
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


async def _ingest_document(
    session: AsyncSession,
    *,
    client: httpx.Client,
    discovered: DiscoveredDocument,
    previous: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[FetchRunResult, ParseRunResult, ChunkRunResult]:
    """Fetch, parse and chunk one document, stopping at the first stage that could not."""
    fetched, fetch_result = await fetch_document(
        session, client=client, discovered=discovered, previous=previous, run=run, store=store
    )
    if fetched is None:
        return fetch_result, ParseRunResult(), ChunkRunResult()
    parsed, parse_result = parse_document(fetched)
    if parsed is None:
        return fetch_result, parse_result, ChunkRunResult()
    chunk_result = await chunk_and_store_document(session, parsed, ingest_run_id=run.id)
    return fetch_result, parse_result, chunk_result


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
        result.mark_reported("discover")
        logger.info("[discover] %s", result.discover.summary())

        result.mark_reported("fetch", "parse", "chunk")
        for document in discovered:
            fetch_result, parse_result, chunk_result = await _ingest_document(
                session,
                client=client,
                discovered=document,
                previous=previous.get(document.celex),
                run=run,
                store=store,
            )
            await session.commit()
            result.fetch.merge(fetch_result)
            result.parse.merge(parse_result)
            result.chunk.merge(chunk_result)

        logger.info("[fetch] %s", result.fetch.summary())
        logger.info("[parse] %s", result.parse.summary())

        corpus_celexes = await celexes_to_keep(
            session, discovered=discovered, result=result, topics=topics
        )
        if corpus_celexes is not None:
            result.chunk += await prune_chunks(session, corpus_celexes=corpus_celexes)
        logger.info("[chunk] %s", result.chunk.summary())

        result.embed = await embed_chunks(session)
        result.mark_reported("embed")
        logger.info("[embed] %s", result.embed.summary())

    return result
