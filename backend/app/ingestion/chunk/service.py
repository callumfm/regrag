"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, select, update
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
    chunk: Chunk, *, digest: str, occurrence: int, corpus_version: str | None
) -> DocumentChunk:
    """The chunk itself, plus what only persistence knows: hash, duplicate index, version."""
    return DocumentChunk(
        **chunk.model_dump(mode="json"),
        content_hash=digest,
        occurrence=occurrence,
        corpus_version=corpus_version,
    )


async def _stamp_unversioned(
    session: AsyncSession, row_ids: Collection[int], corpus_version: str
) -> None:
    """Backfill rows a partial run left unstamped, now that a real corpus version exists."""
    await session.execute(
        update(DocumentChunk)
        .where(DocumentChunk.id.in_(row_ids), DocumentChunk.corpus_version.is_(None))
        .values(corpus_version=corpus_version)
    )


async def upsert_document_chunks(
    session: AsyncSession, *, ref: str, chunks: Sequence[Chunk], corpus_version: str | None
) -> ChunkRunResult:
    """Reconcile a document's chunks by content hash, leaving matched rows otherwise untouched."""
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
    matched = existing.keys() & incoming.keys()
    added = [key for key in incoming if key not in existing]
    if gone:
        await session.execute(
            delete(DocumentChunk).where(DocumentChunk.id.in_([existing[key] for key in gone]))
        )
    if matched and corpus_version is not None:
        await _stamp_unversioned(session, [existing[key] for key in matched], corpus_version)
    session.add_all(
        to_chunk_row(incoming[key], digest=key[0], occurrence=key[1], corpus_version=corpus_version)
        for key in added
    )
    await session.flush()
    return ChunkRunResult(added=len(added), removed=len(gone), unchanged=len(matched))


async def delete_chunks_outside(session: AsyncSession, *, keep_refs: Collection[str]) -> int:
    """Drop chunks of documents no topic holds; the keep list decides, not the topic tag."""
    if not keep_refs:
        return 0
    result = await session.execute(delete(DocumentChunk).where(DocumentChunk.ref.notin_(keep_refs)))
    await session.flush()
    return cast(CursorResult, result).rowcount
