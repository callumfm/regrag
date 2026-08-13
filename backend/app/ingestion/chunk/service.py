"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, cast

from sqlalchemy import CursorResult, Row, delete, func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.models import Chunk, ChunkCounts
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.exceptions import EmptyChunkSetError

ContentKey = tuple[str, int]
"""What identifies a chunk within its document: content hash, then occurrence."""


class ChunkToEmbed(NamedTuple):
    """One vectorless chunk: everything the embed stage needs to fill its vector in."""

    id: int
    celex: str
    text: str


def _key_incoming_chunks(chunks: Iterable[Chunk]) -> dict[ContentKey, Chunk]:
    """Freshly chunked text keyed by content hash and occurrence separating identical siblings."""
    seen: Counter[str] = Counter()
    keyed: dict[ContentKey, Chunk] = {}
    for chunk in chunks:
        digest = chunk.content_hash
        keyed[digest, seen[digest]] = chunk
        seen[digest] += 1
    return keyed


async def _key_stored_chunks(session: AsyncSession, celex: str) -> dict[ContentKey, Row[Any]]:
    """Existing chunked text keyed by content hash and occurrence separating identical siblings."""
    stmt = select(
        DocumentChunk.content_hash,
        DocumentChunk.occurrence,
        DocumentChunk.id,
        *(getattr(DocumentChunk, column) for column in sorted(Chunk.NOT_IDENTITY)),
    ).where(DocumentChunk.celex == celex)
    rows = await session.execute(stmt)
    return {(row.content_hash, row.occurrence): row for row in rows}


async def create_chunks(
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


async def delete_chunks(session: AsyncSession, chunk_ids: Collection[int]) -> int:
    """Drop chunk rows by id, returning how many went."""
    if not chunk_ids:
        return 0

    stmt = delete(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
    result = await session.execute(stmt)
    return cast(CursorResult, result).rowcount


async def prune_chunks(session: AsyncSession, celexes_to_keep: Collection[str]) -> int:
    """Drop the chunks no topic wants anymore."""
    if not celexes_to_keep:
        return 0

    stmt = delete(DocumentChunk).where(DocumentChunk.celex.notin_(celexes_to_keep))
    result = await session.execute(stmt)
    return cast(CursorResult, result).rowcount


async def update_chunks(session: AsyncSession, updates: Sequence[dict[str, Any]]) -> None:
    """Bulk-update chunk rows; each dict carries an id plus the columns to set."""
    if not updates:
        return
    stmt = update(DocumentChunk)
    await session.execute(stmt, updates)


def _updates_for_changed_chunks(
    matched: Collection[ContentKey],
    incoming: Mapping[ContentKey, Chunk],
    existing: Mapping[ContentKey, Row[Any]],
) -> list[dict[str, Any]]:
    """Update payloads for matched rows whose derived columns drifted from the chunker's output."""
    updates = []
    for key in matched:
        row = existing[key]
        new = incoming[key].model_dump(mode="json", include=Chunk.NOT_IDENTITY)
        old = {column: getattr(row, column) for column in Chunk.NOT_IDENTITY}
        if new != old:
            updates.append({"id": row.id, **new})
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
    await create_chunks(session, added, ingest_run_id=ingest_run_id)

    matched = existing.keys() & incoming.keys()
    updates = _updates_for_changed_chunks(matched, incoming, existing)
    await update_chunks(session, updates)

    return ChunkCounts(
        added=len(added),
        deleted=len(deleted),
        kept=len(matched) - len(updates),
        updated=len(updates),
    )


async def get_unembedded_chunks(
    session: AsyncSession, *, after: tuple[str, int] | None = None, limit: int
) -> Sequence[ChunkToEmbed]:
    """One page of vectorless chunks, ordered so a batch can stay inside one document.

    Keyset, not OFFSET: the sweep writes vectors as it reads, so rows leave this query's
    predicate mid-run and an offset would skip past the ones that shifted under it.
    """
    stmt = (
        select(DocumentChunk.id, DocumentChunk.celex, DocumentChunk.text)
        .where(DocumentChunk.embedding.is_(None))
        .order_by(DocumentChunk.celex, DocumentChunk.id)
        .limit(limit)
    )
    if after is not None:
        stmt = stmt.where(tuple_(DocumentChunk.celex, DocumentChunk.id) > after)
    return [ChunkToEmbed(*row) for row in await session.execute(stmt)]


async def set_chunk_embeddings(
    session: AsyncSession, vectors: Mapping[int, Sequence[float]]
) -> None:
    """Write chunk vectors in one executemany UPDATE by primary key, fetching nothing back."""
    if not vectors:
        return
    stmt = update(DocumentChunk)
    await session.execute(
        stmt, [{"id": chunk_id, "embedding": vector} for chunk_id, vector in vectors.items()]
    )


async def count_embedded_chunks(session: AsyncSession) -> int:
    """How many chunks already carry a vector."""
    stmt = (
        select(func.count()).select_from(DocumentChunk).where(DocumentChunk.embedding.is_not(None))
    )
    return await session.scalar(stmt) or 0
