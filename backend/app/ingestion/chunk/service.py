"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.models import Chunk, ChunkRunResult
from app.ingestion.chunk.schemas import DocumentChunk


def keyed(chunks: Iterable[Chunk]) -> Iterator[tuple[Chunk, str, int]]:
    """Pair each chunk with its hash and the occurrence disambiguating identical siblings."""
    seen: Counter[str] = Counter()
    for chunk in chunks:
        digest = chunk.content_hash
        yield chunk, digest, seen[digest]
        seen[digest] += 1


def to_chunk_row(
    chunk: Chunk, *, digest: str, occurrence: int, corpus_version: str
) -> DocumentChunk:
    """The chunk itself, plus what only persistence knows: hash, duplicate index, version."""
    return DocumentChunk(
        **chunk.model_dump(mode="json"),
        content_hash=digest,
        occurrence=occurrence,
        corpus_version=corpus_version,
    )


async def upsert_document_chunks(
    session: AsyncSession, *, ref: str, chunks: Sequence[Chunk], corpus_version: str
) -> ChunkRunResult:
    """Reconcile a document's chunks by content hash, leaving matched rows untouched."""
    incoming = {(digest, occurrence): chunk for chunk, digest, occurrence in keyed(chunks)}
    existing = {
        (content_hash, occurrence): row_id
        for row_id, content_hash, occurrence in await session.execute(
            select(DocumentChunk.id, DocumentChunk.content_hash, DocumentChunk.occurrence).where(
                DocumentChunk.ref == ref
            )
        )
    }
    gone = existing.keys() - incoming.keys()
    added = [key for key in incoming if key not in existing]
    if gone:
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.id.in_([existing[key] for key in gone]))
        )
    session.add_all(
        to_chunk_row(incoming[key], digest=key[0], occurrence=key[1], corpus_version=corpus_version)
        for key in added
    )
    await session.flush()
    return ChunkRunResult(
        added=len(added), removed=len(gone), unchanged=len(existing.keys() & incoming.keys())
    )


async def delete_chunks_outside(
    session: AsyncSession, *, topics: Sequence[str], discovered_refs: Collection[str]
) -> int:
    """Drop chunks of documents no longer discovered for the topics being ingested."""
    if not topics or not discovered_refs:
        return 0
    result = await session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.topic.in_(topics), DocumentChunk.ref.notin_(discovered_refs)
        )
    )
    await session.flush()
    return cast(CursorResult, result).rowcount
