"""Chunk stage: parsed documents to reconciled document_chunks rows."""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.models import ChunkCounts
from app.ingestion.chunk.service import upsert_document_chunks
from app.ingestion.enums import Stage
from app.ingestion.exceptions import DocumentFailed, IngestionError
from app.ingestion.parse.models import ParsedDocument


async def chunk_and_store_document(
    session: AsyncSession, document: ParsedDocument, *, ingest_run_id: int
) -> ChunkCounts:
    """Reconcile one document's chunks against the rows already stored for it."""
    try:
        chunks = chunk_document(document)
        return await upsert_document_chunks(
            session,
            celex=document.celex,
            chunks=chunks,
            ingest_run_id=ingest_run_id,
        )
    except (IngestionError, SQLAlchemyError) as exc:
        raise DocumentFailed(Stage.CHUNK, document.celex, exc) from exc
