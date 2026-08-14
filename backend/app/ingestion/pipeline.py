"""The ingest pipeline: one run, discover -> fetch -> parse -> chunk -> embed."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import ObjectStore
from app.ingestion.chunk.service import prune_chunks
from app.ingestion.chunk.stage import chunk_and_store_document
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.stage import discover_topics, find_dropped_celexes
from app.ingestion.embed.stage import embed_chunks
from app.ingestion.enums import IngestRunStatus, Stage
from app.ingestion.exceptions import DocumentFailed
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.stage import fetch_document, previous_corpus
from app.ingestion.models import DocumentOutcome, IngestRunResult
from app.ingestion.parse.stage import parse_document
from app.ingestion.schemas import IngestRun
from app.ingestion.service import complete_ingest_run, create_ingest_run, get_celexes_to_keep

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _recorded_run(
    session: AsyncSession,
) -> AsyncIterator[tuple[IngestRun, IngestRunResult]]:
    """Open a run and close it out however the body ends, marking it aborted on a raise."""
    run = await create_ingest_run(session)
    result = IngestRunResult(run_id=run.id)
    try:
        yield run, result
        _ = await complete_ingest_run(session, run, status=result.status, result=result.report())
        result.corpus_version = run.corpus_version
    except BaseException:
        await _mark_aborted(session, run, result)
        raise


async def _mark_aborted(session: AsyncSession, run: IngestRun, result: IngestRunResult) -> None:
    """Discard the aborted transaction, then close the run out without masking why it aborted."""
    run_id = run.id
    await session.rollback()
    try:
        _ = await complete_ingest_run(
            session, run, status=IngestRunStatus.ABORTED, result=result.report()
        )
    except SQLAlchemyError:
        logger.exception("run %s could not be marked aborted", run_id)


async def _ingest_document(
    session: AsyncSession,
    document: DiscoveredDocument,
    *,
    previous: RawDocument | None,
    client: httpx.AsyncClient,
    run: IngestRun,
    store: ObjectStore,
) -> DocumentOutcome:
    """One document through fetch -> parse -> chunk: committed whole, or rolled back as failed."""
    try:
        async with session.begin_nested():
            fetched = await fetch_document(
                session, client=client, discovered=document, previous=previous, run=run, store=store
            )
            parsed = parse_document(fetched.raw, fetched.html)
            chunks = await chunk_and_store_document(session, parsed, ingest_run_id=run.id)
    except DocumentFailed as failure:
        return DocumentOutcome(celex=failure.celex, failed=failure.stage, error=failure.reason)
    await session.commit()
    return DocumentOutcome(celex=document.celex, change=fetched.change, chunks=chunks)


async def ingest(
    session: AsyncSession,
    *,
    client: httpx.AsyncClient,
    topics: Sequence[str],
    store: ObjectStore,
) -> IngestRunResult:
    """Run the whole pipeline under one ingest run, each document landing whole or not at all."""
    async with _recorded_run(session) as (run, result):
        existing = await previous_corpus(session, topics)
        discovered = await discover_topics(client, topics)
        result.discovered = len(discovered)
        result.dropped = find_dropped_celexes(discovered, existing.keys())
        logger.info("%s", result.line(Stage.DISCOVER))

        for document in discovered:
            outcome = await _ingest_document(
                session,
                document,
                previous=existing.get(document.celex),
                client=client,
                run=run,
                store=store,
            )
            result.documents.append(outcome)

        logger.info("%s", result.line(Stage.FETCH))
        logger.info("%s", result.line(Stage.PARSE))

        if result.corpus_complete:
            to_keep = await get_celexes_to_keep(session, discovered=discovered, topics=topics)
            result.pruned = await prune_chunks(session, to_keep)

        logger.info("%s", result.line(Stage.CHUNK))

        result.embed = await embed_chunks(session)
        logger.info("%s", result.line(Stage.EMBED))
    return result
