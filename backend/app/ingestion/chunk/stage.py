"""Chunk stage: parsed documents to reconciled document_chunks rows."""

from collections.abc import Collection, Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.chunk.service import delete_chunks_outside, upsert_document_chunks
from app.ingestion.exceptions import IngestionError
from app.ingestion.parse.models import ParsedDocument


async def chunk_and_store_documents(
    session: AsyncSession,
    documents: Sequence[ParsedDocument],
    *,
    ingest_run_id: int,
) -> ChunkRunResult:
    """Reconcile each document's chunks against the rows already stored for it."""
    result = ChunkRunResult()
    for document in documents:
        try:
            async with session.begin_nested():
                result += await upsert_document_chunks(
                    session,
                    celex=document.celex,
                    chunks=chunk_document(document),
                    ingest_run_id=ingest_run_id,
                )
        except (IngestionError, SQLAlchemyError) as exc:
            result.fail(document.celex, exc)
    return result


async def prune_chunks(session: AsyncSession, *, corpus_celexes: Collection[str]) -> ChunkRunResult:
    """Drop the chunks of every celex the corpus no longer holds."""
    removed = await delete_chunks_outside(session, corpus_celexes=corpus_celexes)
    return ChunkRunResult(removed=removed)
