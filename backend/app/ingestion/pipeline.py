"""The ingest pipeline: one run, fetch -> parse -> chunk -> store."""

import logging
from collections.abc import Sequence
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.service import delete_chunks_for_refs, upsert_document_chunks
from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import DiscoveryError, ParseError
from app.ingestion.fetch.corpus import fetch_documents
from app.ingestion.models import RunReport
from app.ingestion.parse.eurlex_html import parse_eurlex_html
from app.ingestion.schemas import IngestedDocument
from app.ingestion.service import complete_ingest_run, create_ingest_run, next_corpus_version

logger = logging.getLogger(__name__)


async def chunk_documents(
    session: AsyncSession,
    documents: Sequence[IngestedDocument],
    data_dir: Path,
    corpus_version: str,
    report: RunReport,
) -> int:
    """Parse and chunk each fetched document, reconciling its rows; returns chunks produced."""
    produced = 0
    for document in documents:
        try:
            parsed = parse_eurlex_html(
                (data_dir / f"{document.ref}.html").read_text(), document.ref, document.topic
            )
        except (ParseError, OSError) as exc:
            report.unparsed[document.ref] = f"{type(exc).__name__}: {exc}"
            continue
        chunks = chunk_document(parsed)
        produced += len(chunks)
        delta = await upsert_document_chunks(session, document.ref, chunks, corpus_version)
        report.chunks_added += delta.added
        report.chunks_removed += delta.removed
        report.chunks_unchanged += delta.unchanged
    return produced


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
        documents = await fetch_documents(session, client, topics, data_dir, run, report)
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
    version = await next_corpus_version(session)
    report.chunks_removed += await delete_chunks_for_refs(session, report.dropped)
    produced = await chunk_documents(session, documents, data_dir, version, report)
    logger.info("[chunk] %d documents -> %d chunks", len(documents), produced)
    logger.info(
        "[store] +%d added, -%d removed, %d unchanged",
        report.chunks_added,
        report.chunks_removed,
        report.chunks_unchanged,
    )
    report.corpus_version = version if report.ok else None
    await complete_ingest_run(session, run, report.status, report.corpus_version)
    return report
