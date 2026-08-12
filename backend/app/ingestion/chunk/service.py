"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, cast

from sqlalchemy import CursorResult, delete, func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.ingestion.chunk.models import Chunk, ChunkCounts
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.exceptions import EmptyChunkSetError

ContentKey = tuple[str, int]
"""What identifies a chunk within its document: content hash, then occurrence."""


class StoredChunk(NamedTuple):
    """What reconciliation reads off a stored row: its id and its derived fields."""

    id: int
    topic: str
    citation: str
    references: list[dict]


def derived_values(chunk: Chunk) -> dict[str, Any]:
    """Columns code derives from hashed content, as the row would store them."""
    return chunk.model_dump(mode="json", include=Chunk.NOT_IDENTITY)


def stored_values(stored: StoredChunk) -> dict[str, Any]:
    """A row's derived columns in the same shape, for comparison."""
    return {key: value for key, value in stored._asdict().items() if key != "id"}


def stale_updates(
    matched: Collection[ContentKey],
    incoming: Mapping[ContentKey, Chunk],
    existing: Mapping[ContentKey, StoredChunk],
) -> list[dict[str, Any]]:
    """Updates for matched rows whose derived fields no longer match what the chunker produced."""
    updates = []
    for key in matched:
        values = derived_values(incoming[key])
        if values != stored_values(existing[key]):
            updates.append({"id": existing[key].id, **values})
    return updates


def key_incoming_chunks(chunks: Iterable[Chunk]) -> dict[ContentKey, Chunk]:
    """Freshly chunked text keyed by content hash, occurrence separating identical siblings."""
    seen: Counter[str] = Counter()
    keyed: dict[ContentKey, Chunk] = {}
    for chunk in chunks:
        digest = chunk.content_hash
        keyed[digest, seen[digest]] = chunk
        seen[digest] += 1
    return keyed


async def key_stored_chunks(session: AsyncSession, celex: str) -> dict[ContentKey, StoredChunk]:
    """A document's stored rows under the same keys the incoming chunks carry."""
    stmt = select(
        DocumentChunk.content_hash,
        DocumentChunk.occurrence,
        DocumentChunk.id,
        DocumentChunk.topic,
        DocumentChunk.citation,
        DocumentChunk.references,
    ).where(DocumentChunk.celex == celex)
    rows = await session.execute(stmt)
    chunks = {
        (row.content_hash, row.occurrence): StoredChunk(
            row.id, row.topic, row.citation, row.references
        )
        for row in rows
    }
    return chunks


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


async def refresh_chunks(session: AsyncSession, updates: Sequence[dict[str, Any]]) -> None:
    """Rewrite derived fields on matched rows, leaving text and embedding untouched."""
    if not updates:
        return
    stmt = update(DocumentChunk)
    await session.execute(stmt, updates)
    await session.flush()


async def upsert_document_chunks(
    session: AsyncSession, *, celex: str, chunks: Sequence[Chunk], ingest_run_id: int
) -> ChunkCounts:
    """Reconcile a document's chunks by content hash, refreshing derived fields on matches."""
    incoming = key_incoming_chunks(chunks)
    existing = await key_stored_chunks(session, celex)
    if not incoming and existing:
        raise EmptyChunkSetError(f"{celex}: chunked to nothing over {len(existing)} stored chunks")

    gone = [existing[key].id for key in existing.keys() - incoming.keys()]
    await delete_chunks(session, gone)

    added = {key: chunk for key, chunk in incoming.items() if key not in existing}
    await insert_chunks(session, added, ingest_run_id=ingest_run_id)

    matched = existing.keys() & incoming.keys()
    stale = stale_updates(matched, incoming, existing)
    await refresh_chunks(session, stale)

    return ChunkCounts(
        added=len(added),
        deleted=len(gone),
        kept=len(matched) - len(stale),
        refreshed=len(stale),
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
