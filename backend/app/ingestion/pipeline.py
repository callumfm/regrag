"""The ingest pipeline: one run, fetch -> parse -> chunk -> store."""

import logging
from collections.abc import Sequence
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.models import ChunkDelta
from app.ingestion.chunk.service import delete_chunks_outside, upsert_document_chunks
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
) -> tuple[int, int]:
    """Parse and chunk each fetched document, reconciling its rows; returns (documents, chunks)."""
    chunked = 0
    produced = 0
    for document in documents:
        try:
            parsed = parse_eurlex_html(
                (data_dir / f"{document.ref}.html").read_text(encoding="utf-8"),
                document.ref,
                document.topic,
            )
        except (ParseError, OSError, UnicodeDecodeError) as exc:
            report.parse.failed[document.ref] = f"{type(exc).__name__}: {exc}"
            continue
        report.parse.parsed.append(document.ref)
        chunks = chunk_document(parsed)
        chunked += 1
        produced += len(chunks)
        delta = await upsert_document_chunks(session, document.ref, chunks, corpus_version)
        report.chunk += delta
    return chunked, produced


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
        documents = await fetch_documents(session, client, topics, data_dir, run, report.fetch)
    except (DiscoveryError, httpx.HTTPError):
        await complete_ingest_run(session, run, IngestRunStatus.FAILED)
        raise
    logger.info("[fetch] %s", report.fetch.summary())
    version = await next_corpus_version(session)
    report.chunk += ChunkDelta(
        removed=await delete_chunks_outside(session, topics, report.fetch.discovered)
    )
    chunked, produced = await chunk_documents(session, documents, data_dir, version, report)
    logger.info("[chunk] %d documents -> %d chunks", chunked, produced)
    logger.info("[chunk] %s", report.chunk.summary())
    report.corpus_version = version if report.ok else None
    await complete_ingest_run(session, run, report.status, report.corpus_version)
    return report
