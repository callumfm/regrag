"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, Row, delete, func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.ingestion.chunk.models import Chunk, ChunkCounts
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.exceptions import EmptyChunkSetError

ContentKey = tuple[str, int]
"""What identifies a chunk within its document: content hash, then occurrence."""


def _key_incoming_chunks(chunks: Iterable[Chunk]) -> dict[ContentKey, Chunk]:
    """Freshly chunked text keyed by content hash, occurrence separating identical siblings."""
    seen: Counter[str] = Counter()
    keyed: dict[ContentKey, Chunk] = {}
    for chunk in chunks:
        digest = chunk.content_hash
        keyed[digest, seen[digest]] = chunk
        seen[digest] += 1
    return keyed


async def _key_stored_chunks(session: AsyncSession, celex: str) -> dict[ContentKey, Row[Any]]:
    """A document's stored rows (id plus derived columns) under the same keys
    incoming chunks carry."""
    stmt = select(
        DocumentChunk.content_hash,
        DocumentChunk.occurrence,
        DocumentChunk.id,
        *(getattr(DocumentChunk, column) for column in sorted(Chunk.NOT_IDENTITY)),
    ).where(DocumentChunk.celex == celex)
    rows = await session.execute(stmt)
    return {(row.content_hash, row.occurrence): row for row in rows}


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


async def update_chunks(session: AsyncSession, updates: Sequence[dict[str, Any]]) -> None:
    """Bulk-update chunk rows; each dict carries an id plus the columns to set."""
    if not updates:
        return
    stmt = update(DocumentChunk)
    await session.execute(stmt, updates)
    await session.flush()


def _stale_chunk_updates(
    matched: Collection[ContentKey],
    incoming: Mapping[ContentKey, Chunk],
    existing: Mapping[ContentKey, Row[Any]],
) -> list[dict[str, Any]]:
    """Update payloads for matched rows whose derived columns drifted from the chunker's output."""
    updates = []
    for key in matched:
        new = incoming[key].model_dump(mode="json", include=Chunk.NOT_IDENTITY)
        old = {column: getattr(existing[key], column) for column in Chunk.NOT_IDENTITY}
        if new != old:
            updates.append({"id": existing[key].id, **new})
    return updates


async def sync_document_chunks(
    session: AsyncSession, *, celex: str, chunks: Sequence[Chunk], ingest_run_id: int
) -> ChunkCounts:
    """Make a document's stored chunks match this set: insert new, delete gone, update drifted."""
    incoming = _key_incoming_chunks(chunks)
    existing = await _key_stored_chunks(session, celex)
    if not incoming and existing:
        raise EmptyChunkSetError(f"{celex}: chunked to nothing over {len(existing)} stored chunks")

    deleted = [existing[key].id for key in existing.keys() - incoming.keys()]
    await delete_chunks(session, deleted)

    added = {key: chunk for key, chunk in incoming.items() if key not in existing}
    await insert_chunks(session, added, ingest_run_id=ingest_run_id)

    matched = existing.keys() & incoming.keys()
    updates = _stale_chunk_updates(matched, incoming, existing)
    await update_chunks(session, updates)

    return ChunkCounts(
        added=len(added),
        deleted=len(deleted),
        kept=len(matched) - len(updates),
        refreshed=len(updates),
    )


async def get_chunks(
    session: AsyncSession,
    *,
    has_embedding: bool | None = None,
    after: tuple[str, int] | None = None,
    limit: int | None = None,
) -> Sequence[DocumentChunk]:
    """Chunks ordered by (celex, id); `after` pages by keyset because the embed sweep
    moves rows out of the filter mid-scan, which would shift an OFFSET under it."""
    stmt = (
        select(DocumentChunk)
        .options(defer(DocumentChunk.search_vector))
        .order_by(DocumentChunk.celex, DocumentChunk.id)
    )
    if has_embedding is not None:
        stmt = stmt.where(
            DocumentChunk.embedding.is_not(None)
            if has_embedding
            else DocumentChunk.embedding.is_(None)
        )
    if after is not None:
        stmt = stmt.where(tuple_(DocumentChunk.celex, DocumentChunk.id) > after)
    if limit is not None:
        stmt = stmt.limit(limit)
    return (await session.scalars(stmt)).all()


async def count_chunks(session: AsyncSession, *, has_embedding: bool | None = None) -> int:
    """How many chunks there are, optionally filtered by whether they carry a vector."""
    stmt = select(func.count()).select_from(DocumentChunk)
    if has_embedding is not None:
        stmt = stmt.where(
            DocumentChunk.embedding.is_not(None)
            if has_embedding
            else DocumentChunk.embedding.is_(None)
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
