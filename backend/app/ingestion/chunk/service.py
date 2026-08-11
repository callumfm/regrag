"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from typing import cast

from sqlalchemy import CursorResult, delete, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.ingestion.chunk.models import Chunk, ChunkRunResult
from app.ingestion.chunk.schemas import DocumentChunk

ContentKey = tuple[str, int]
"""What identifies a chunk within its document: content hash, then occurrence."""


def with_content_keys(chunks: Iterable[Chunk]) -> Iterator[tuple[Chunk, str, int]]:
    """Pair each chunk with its hash and the occurrence disambiguating identical siblings."""
    seen: Counter[str] = Counter()
    for chunk in chunks:
        digest = chunk.content_hash
        yield chunk, digest, seen[digest]
        seen[digest] += 1


async def get_chunk_ids(session: AsyncSession, celex: str) -> dict[ContentKey, int]:
    """Row ids of a document's stored chunks, keyed by content hash and occurrence."""
    stmt = select(DocumentChunk.content_hash, DocumentChunk.occurrence, DocumentChunk.id).where(
        DocumentChunk.celex == celex
    )
    rows = await session.execute(stmt)
    return {(content_hash, occurrence): row_id for content_hash, occurrence, row_id in rows}


async def insert_chunks(
    session: AsyncSession, chunks: Mapping[ContentKey, Chunk], *, ingest_run_id: int
) -> None:
    """Store chunks under their content keys, adding what only persistence knows."""
    session.add_all(
        DocumentChunk(
            **chunk.model_dump(mode="json"),
            content_hash=digest,
            occurrence=occurrence,
            ingest_run_id=ingest_run_id,
        )
        for (digest, occurrence), chunk in chunks.items()
    )
    await session.flush()


async def delete_chunks(session: AsyncSession, chunk_ids: Collection[int]) -> None:
    """Drop chunk rows by id."""
    if not chunk_ids:
        return
    stmt = delete(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
    await session.execute(stmt)
    await session.flush()


async def upsert_document_chunks(
    session: AsyncSession, *, celex: str, chunks: Sequence[Chunk], ingest_run_id: int
) -> ChunkRunResult:
    """Reconcile a document's chunks by content hash, leaving matched rows otherwise untouched."""
    incoming = {(digest, n): chunk for chunk, digest, n in with_content_keys(chunks)}
    existing = await get_chunk_ids(session, celex)
    gone = existing.keys() - incoming.keys()
    added = [key for key in incoming if key not in existing]
    await delete_chunks(session, [existing[key] for key in gone])
    await insert_chunks(session, {key: incoming[key] for key in added}, ingest_run_id=ingest_run_id)
    return ChunkRunResult(
        added=len(added), removed=len(gone), unchanged=len(existing.keys() & incoming.keys())
    )


async def get_unembedded_chunks(
    session: AsyncSession, *, after: tuple[str, int] | None = None, limit: int
) -> Sequence[DocumentChunk]:
    """One page of vectorless chunks, ordered so a batch can stay inside one document.

    Keyset, not OFFSET: the sweep writes vectors as it reads, so rows leave this query's
    predicate mid-run and an offset would skip past the ones that shifted under it.
    """
    stmt = (
        select(DocumentChunk)
        .options(defer(DocumentChunk.search_vector))
        .where(DocumentChunk.embedding.is_(None))
        .order_by(DocumentChunk.celex, DocumentChunk.id)
        .limit(limit)
    )
    if after is not None:
        stmt = stmt.where(tuple_(DocumentChunk.celex, DocumentChunk.id) > after)
    return (await session.scalars(stmt)).all()


async def count_embedded_chunks(session: AsyncSession) -> int:
    """How many chunks already carry a vector."""
    stmt = (
        select(func.count()).select_from(DocumentChunk).where(DocumentChunk.embedding.is_not(None))
    )
    return await session.scalar(stmt) or 0


async def delete_chunks_outside(session: AsyncSession, *, corpus_celexes: Collection[str]) -> int:
    """Drop chunks of documents no topic holds; the corpus decides, not the topic tag."""
    if not corpus_celexes:
        return 0
    stmt = delete(DocumentChunk).where(DocumentChunk.celex.notin_(corpus_celexes))
    result = await session.execute(stmt)
    await session.flush()
    return cast(CursorResult, result).rowcount
