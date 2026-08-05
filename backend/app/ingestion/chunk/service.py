"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections.abc import Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import Chunk
from app.ingestion.chunk.identity import keyed
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.models import ChunkDelta


def to_chunk_row(chunk: Chunk, digest: str, occurrence: int, corpus_version: str) -> DocumentChunk:
    """Map the chunker's value object onto its persisted row."""
    return DocumentChunk(
        ref=chunk.ref,
        topic=chunk.topic,
        content_hash=digest,
        occurrence=occurrence,
        kind=chunk.kind,
        article=chunk.article,
        annex=chunk.annex,
        title=chunk.title,
        paragraph=chunk.paragraph,
        heading_path=list(chunk.heading_path),
        part=chunk.part,
        parts=chunk.parts,
        citation=chunk.citation,
        text=chunk.text,
        references=[reference.model_dump() for reference in chunk.references],
        corpus_version=corpus_version,
    )


async def upsert_document_chunks(
    session: AsyncSession, ref: str, chunks: Sequence[Chunk], corpus_version: str
) -> ChunkDelta:
    """Reconcile a document's chunks by content hash, leaving matched rows untouched."""
    incoming = {(digest, occurrence): chunk for chunk, digest, occurrence in keyed(chunks)}
    existing = {
        (row.content_hash, row.occurrence): row
        for row in await session.scalars(select(DocumentChunk).where(DocumentChunk.ref == ref))
    }
    gone = existing.keys() - incoming.keys()
    added = [key for key in incoming if key not in existing]
    for key in gone:
        await session.delete(existing[key])
    session.add_all(to_chunk_row(incoming[key], key[0], key[1], corpus_version) for key in added)
    await session.flush()
    return ChunkDelta(
        added=len(added), removed=len(gone), unchanged=len(existing.keys() & incoming.keys())
    )


async def delete_chunks_for_refs(session: AsyncSession, refs: Sequence[str]) -> int:
    """Drop every chunk of documents no longer in the corpus."""
    if not refs:
        return 0
    result = await session.execute(delete(DocumentChunk).where(DocumentChunk.ref.in_(refs)))
    await session.flush()
    return cast(CursorResult, result).rowcount
