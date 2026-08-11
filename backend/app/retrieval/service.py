"""Retrieval reads: the two candidate legs, and the exact lookups over stored chunks."""

import re
from collections.abc import Sequence

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import RetrievedChunk, SearchFilters


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


PARAGRAPH_NUMBER = re.compile(r"(\d+)(.*)")

CHUNK_COLUMNS = (
    DocumentChunk.id,
    DocumentChunk.celex,
    DocumentChunk.topic,
    DocumentChunk.citation,
    DocumentChunk.title,
    DocumentChunk.text,
)
"""What a caller sees; never the embedding or the search vector, which are large and internal."""


def natural_key(paragraph: str | None) -> tuple[int, str]:
    """Sort key for a paragraph: chapeau first, then '2' before '10', '11' before '11a'."""
    if paragraph is None:
        return (-1, "")
    match = PARAGRAPH_NUMBER.match(paragraph)
    return (int(match[1]), match[2]) if match else (0, paragraph)


async def hydrate(session: AsyncSession, chunk_ids: Sequence[int]) -> dict[int, RetrievedChunk]:
    """The chunks behind a set of ids, keyed by id because a query cannot preserve their order."""
    if not chunk_ids:
        return {}
    stmt = select(*CHUNK_COLUMNS).where(DocumentChunk.id.in_(chunk_ids))
    rows = await session.execute(stmt)
    return {row.id: RetrievedChunk.model_validate(row, from_attributes=True) for row in rows}


async def get_article(
    session: AsyncSession, *, celex: str, article: str
) -> tuple[RetrievedChunk, ...]:
    """One article's chunks in reading order; an article the act does not have returns nothing."""
    stmt = select(*CHUNK_COLUMNS, DocumentChunk.paragraph, DocumentChunk.part).where(
        DocumentChunk.celex == celex,
        func.lower(DocumentChunk.article) == article.lower(),
    )
    rows = sorted(
        await session.execute(stmt), key=lambda row: (natural_key(row.paragraph), row.part)
    )
    return tuple(RetrievedChunk.model_validate(row, from_attributes=True) for row in rows)
