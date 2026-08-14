"""Retrieval reads: the fused hybrid search, and the exact lookups over stored chunks."""

from collections.abc import Sequence

from sqlalchemy import Integer, Select, cast, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import RetrievedChunk, SearchFilters, SearchResult

ITERATIVE_SCAN = text(
    "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
    f" set_config('hnsw.ef_search', '{config.SEARCH_CANDIDATES * 4}', true)"
)
"""Resume the HNSW walk past dead tuples so a limit is met, in the exact order fusion ranks on."""

CHUNK_COLUMNS = (
    DocumentChunk.id,
    DocumentChunk.celex,
    DocumentChunk.topic,
    DocumentChunk.citation,
    DocumentChunk.title,
    DocumentChunk.text,
)

ARTICLE_ORDER = (
    DocumentChunk.paragraph.is_not(None),
    cast(func.substring(DocumentChunk.paragraph, "^[0-9]+"), Integer),
    func.substring(DocumentChunk.paragraph, "^[0-9]*(.*)$"),
    DocumentChunk.part,
)

ANNEX_ORDER = (DocumentChunk.position, DocumentChunk.part)
"""An annex numbers no paragraphs, so where it sits in the document is the only order it has."""


def _filtered(stmt: Select, filters: SearchFilters) -> Select:
    """Narrow a candidate query before its limit, so the pool the fusion sees is honest."""
    if filters.celex is not None:
        stmt = stmt.where(DocumentChunk.celex == filters.celex)
    if filters.topic is not None:
        stmt = stmt.where(DocumentChunk.topic == filters.topic)
    return stmt


def _ranked(stmt: Select, order: Sequence, filters: SearchFilters, limit: int) -> Select:
    """A leg's top ids, each carrying its 1-based position in that leg."""
    return (
        _filtered(stmt, filters)
        .add_columns(func.row_number().over(order_by=order).label("rank"))
        .order_by(*order)
        .limit(limit)
    )


def vector_leg(embedding: Sequence[float], filters: SearchFilters, *, limit: int) -> Select:
    """Chunk ids nearest the query vector by cosine distance, closest first."""
    order = (DocumentChunk.embedding.cosine_distance(embedding), DocumentChunk.id)
    stmt = select(DocumentChunk.id).where(DocumentChunk.embedding.is_not(None))
    return _ranked(stmt, order, filters, limit)


def text_leg(query: str, filters: SearchFilters, *, limit: int) -> Select:
    """Chunk ids whose search vector matches the query, best-ranked first."""
    tsquery = func.websearch_to_tsquery("english", query)
    order = (func.ts_rank_cd(DocumentChunk.search_vector, tsquery).desc(), DocumentChunk.id)
    stmt = select(DocumentChunk.id).where(DocumentChunk.search_vector.bool_op("@@")(tsquery))
    return _ranked(stmt, order, filters, limit)


def _fused(vector: Select, keyword: Select, *, k: int, limit: int) -> Select:
    """Reciprocal Rank Fusion over both legs: 1/(k + rank) summed, best first, id breaking ties.

    A full outer join keeps chunks only one leg found, whose missing term contributes nothing.
    """
    left, right = vector.cte("vector_leg"), keyword.cte("text_leg")
    score = (
        func.coalesce(1.0 / (k + left.c.rank), 0.0) + func.coalesce(1.0 / (k + right.c.rank), 0.0)
    ).label("score")
    joined = left.join(right, left.c.id == right.c.id, full=True).join(
        DocumentChunk, DocumentChunk.id == func.coalesce(left.c.id, right.c.id)
    )
    return (
        select(
            *CHUNK_COLUMNS,
            score,
            left.c.rank.label("vector_rank"),
            right.c.rank.label("text_rank"),
        )
        .select_from(joined)
        .order_by(score.desc(), DocumentChunk.id)
        .limit(limit)
    )


async def hybrid_search(
    session: AsyncSession,
    embedding: Sequence[float],
    query: str,
    filters: SearchFilters,
    *,
    candidates: int,
    k: int,
    limit: int,
) -> tuple[SearchResult, ...]:
    """The corpus's best answers, both legs fused and their chunks loaded in one round trip."""
    await session.execute(ITERATIVE_SCAN)
    stmt = _fused(
        vector_leg(embedding, filters, limit=candidates),
        text_leg(query, filters, limit=candidates),
        k=k,
        limit=limit,
    )
    rows = await session.execute(stmt)
    return tuple(SearchResult.model_validate(row, from_attributes=True) for row in rows)


def _located(
    stmt: Select, *, article: str | None, paragraph: str | None, annex: str | None
) -> Select:
    """Narrow to the division cited: every locator the reference carries is another filter.

    Each is tested against None, not truthiness, since an annex numbered '' is a real annex.
    """
    if article is not None:
        stmt = stmt.where(func.lower(DocumentChunk.article) == article.lower())
    if paragraph is not None:
        stmt = stmt.where(DocumentChunk.paragraph == paragraph)
    if annex is not None:
        stmt = stmt.where(DocumentChunk.annex == annex)
    return stmt


async def follow_reference(
    session: AsyncSession,
    *,
    celex: str,
    article: str | None = None,
    paragraph: str | None = None,
    annex: str | None = None,
) -> tuple[RetrievedChunk, ...]:
    """The text a stored cross-reference points at, in reading order.

    A reference naming an act and nothing else is a filtered search rather than a lookup,
    so it is refused instead of returning the act whole. An act the corpus does not hold
    comes back empty, which the agent can tell the user.
    """
    if article is None and annex is None:
        raise ValueError(f"following {celex} needs an article or annex, not the act alone")
    stmt = _located(
        select(*CHUNK_COLUMNS).where(DocumentChunk.celex == celex),
        article=article,
        paragraph=paragraph,
        annex=annex,
    ).order_by(*(ANNEX_ORDER if annex is not None else ARTICLE_ORDER))
    rows = await session.execute(stmt)
    return tuple(RetrievedChunk.model_validate(row, from_attributes=True) for row in rows)
