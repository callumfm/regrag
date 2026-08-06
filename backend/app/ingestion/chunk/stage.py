"""Chunk stage: parsed documents to reconciled document_chunks rows."""

from collections.abc import Collection, Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.chunk.service import delete_chunks_outside, upsert_document_chunks
from app.ingestion.exceptions import IngestionError
from app.ingestion.parse.models import ParsedDocument


async def chunk_documents(
    session: AsyncSession,
    documents: Sequence[ParsedDocument],
    *,
    corpus_version: str | None,
    keep_refs: Collection[str] | None,
) -> ChunkRunResult:
    """Reconcile each document's chunks; with a keep list, drop chunks of every ref outside it.

    keep_refs is None when the run is incomplete, and an incomplete run prunes nothing.
    """
    result = ChunkRunResult()
    for document in documents:
        try:
            async with session.begin_nested():
                result += await upsert_document_chunks(
                    session,
                    ref=document.ref,
                    chunks=chunk_document(document),
                    corpus_version=corpus_version,
                )
        except (IngestionError, SQLAlchemyError) as exc:
            result.failed[document.ref] = f"{type(exc).__name__}: {exc}"
    if keep_refs is None:
        return result
    dropped = await delete_chunks_outside(session, keep_refs=keep_refs)
    return result + ChunkRunResult(removed=dropped)
