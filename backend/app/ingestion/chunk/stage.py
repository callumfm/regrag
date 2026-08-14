"""Chunk stage: parsed documents to reconciled document_chunks rows."""

from collections.abc import Collection

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.models import ChunkCounts
from app.ingestion.chunk.service import prune_chunks, sync_document_chunks
from app.ingestion.chunk.tree import chunk_document
from app.ingestion.enums import Stage
from app.ingestion.exceptions import DocumentFailed, IngestionError
from app.ingestion.parse.models import ParsedDocument


async def chunk_and_store_document(
    session: AsyncSession, document: ParsedDocument, *, ingest_run_id: int
) -> ChunkCounts:
    """Reconcile one document's chunks against the rows already stored for it."""
    try:
        chunks = chunk_document(document)
        return await sync_document_chunks(
            session,
            celex=document.celex,
            chunks=chunks,
            ingest_run_id=ingest_run_id,
        )
    except (IngestionError, SQLAlchemyError) as exc:
        raise DocumentFailed(Stage.CHUNK, document.celex, exc) from exc


async def prune_dropped_chunks(session: AsyncSession, celexes_to_keep: Collection[str]) -> int:
    """Drop the chunks no topic wants anymore, committed where the deleting happens.

    Left pending, the deletes would ride on whichever later commit fired first and any rollback
    after this point would silently undo them, while the run still reported them as deleted.
    """
    pruned = await prune_chunks(session, celexes_to_keep)
    await session.commit()
    return pruned
