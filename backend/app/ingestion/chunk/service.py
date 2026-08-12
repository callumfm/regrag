"""Chunk persistence: reconcile a document's chunks against what is already stored."""

from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
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


def with_content_keys(chunks: Iterable[Chunk]) -> Iterator[tuple[Chunk, str, int]]:
    """Pair each chunk with its hash and the occurrence disambiguating identical siblings."""
    seen: Counter[str] = Counter()
    for chunk in chunks:
        digest = chunk.content_hash
        yield chunk, digest, seen[digest]
        seen[digest] += 1


async def get_stored_chunks(session: AsyncSession, celex: str) -> dict[ContentKey, StoredChunk]:
    """A document's stored rows, keyed by content hash and occurrence."""
    stmt = select(
        DocumentChunk.content_hash,
        DocumentChunk.occurrence,
        DocumentChunk.id,
        DocumentChunk.topic,
        DocumentChunk.citation,
        DocumentChunk.references,
    ).where(DocumentChunk.celex == celex)
    rows = await session.execute(stmt)
    return {
        (content_hash, occurrence): StoredChunk(row_id, topic, citation, references)
        for content_hash, occurrence, row_id, topic, citation, references in rows
    }


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
    incoming = {(digest, n): chunk for chunk, digest, n in with_content_keys(chunks)}
    existing = await get_stored_chunks(session, celex)
    if not incoming and existing:
        raise EmptyChunkSetError(f"{celex}: chunked to nothing over {len(existing)} stored chunks")
    gone = existing.keys() - incoming.keys()
    added = [key for key in incoming if key not in existing]
    matched = existing.keys() & incoming.keys()
    refreshed = [
        {"id": existing[key].id, **values}
        for key in matched
        if (values := derived_values(incoming[key])) != stored_values(existing[key])
    ]
    await delete_chunks(session, [existing[key].id for key in gone])
    await insert_chunks(session, {key: incoming[key] for key in added}, ingest_run_id=ingest_run_id)
    await refresh_chunks(session, refreshed)
    return ChunkCounts(
        added=len(added),
        deleted=len(gone),
        kept=len(matched) - len(refreshed),
        refreshed=len(refreshed),
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
