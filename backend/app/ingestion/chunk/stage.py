"""Chunk stage: parsed documents to reconciled document_chunks rows."""

from collections.abc import Collection, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.service import delete_chunks_outside, upsert_document_chunks
from app.ingestion.models import ChunkDelta
from app.ingestion.parse.models import ParsedDocument


async def chunk_documents(
    session: AsyncSession,
    documents: Sequence[ParsedDocument],
    *,
    corpus_version: str,
    topics: Sequence[str],
    discovered: Collection[str],
) -> ChunkDelta:
    """Reconcile each document's chunks, then drop chunks of refs no longer discovered."""
    delta = ChunkDelta()
    for document in documents:
        delta += await upsert_document_chunks(
            session,
            ref=document.ref,
            chunks=chunk_document(document),
            corpus_version=corpus_version,
        )
    dropped = await delete_chunks_outside(session, topics=topics, discovered_refs=discovered)
    return delta + ChunkDelta(removed=dropped)
