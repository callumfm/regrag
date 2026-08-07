"""Chunk-vector queries: what still needs embedding, and what already has it."""

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk


async def get_unembedded_chunks(session: AsyncSession) -> Sequence[DocumentChunk]:
    """Every vectorless chunk, ordered so each document's chunks are adjacent.

    Adjacency is load-bearing: the stage groups on it to keep a batch inside one document.
    """
    return (
        await session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.celex, DocumentChunk.id)
        )
    ).all()


async def count_embedded_chunks(session: AsyncSession) -> int:
    """How many chunks already carry a vector."""
    return (
        await session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.embedding.is_not(None))
        )
        or 0
    )
