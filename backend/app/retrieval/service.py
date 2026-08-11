"""Retrieval reads: the two candidate legs, and the exact lookups over stored chunks."""

from collections.abc import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import SearchFilters


def _filtered(stmt: Select, filters: SearchFilters) -> Select:
    """Narrow a candidate query before its limit, so the pool the fusion sees is honest."""
    if filters.celex is not None:
        stmt = stmt.where(DocumentChunk.celex == filters.celex)
    if filters.topic is not None:
        stmt = stmt.where(DocumentChunk.topic == filters.topic)
    return stmt


async def vector_search(
    session: AsyncSession, embedding: Sequence[float], filters: SearchFilters, *, limit: int
) -> list[int]:
    """Chunk ids nearest the query vector by cosine distance, closest first."""
    stmt = (
        select(DocumentChunk.id)
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(DocumentChunk.embedding.cosine_distance(embedding), DocumentChunk.id)
        .limit(limit)
    )
    return list(await session.scalars(_filtered(stmt, filters)))


async def text_search(
    session: AsyncSession, query: str, filters: SearchFilters, *, limit: int
) -> list[int]:
    """Chunk ids whose search vector matches the query, best-ranked first."""
    tsquery = func.websearch_to_tsquery("english", query)
    stmt = (
        select(DocumentChunk.id)
        .where(DocumentChunk.search_vector.bool_op("@@")(tsquery))
        .order_by(func.ts_rank_cd(DocumentChunk.search_vector, tsquery).desc(), DocumentChunk.id)
        .limit(limit)
    )
    return list(await session.scalars(_filtered(stmt, filters)))
