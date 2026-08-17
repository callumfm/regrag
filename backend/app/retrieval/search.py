"""Search the corpus: embed the query, run both legs and fuse their ranks in one query, rerank."""

from collections.abc import Sequence

from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.core.llm import EmbedInput, embed, llm_retry
from app.ingestion.chunk.schemas import DocumentChunk
from app.retrieval.models import CHUNK_COLUMNS, SearchFilters, SearchRequest, SearchResult
from app.retrieval.rerank import rerank_results

EF_SEARCH_MAX = 1000
"""pgvector refuses a larger walk, so a pool that would ask for one is clamped, not rejected."""


@llm_retry
async def _embed_query(query: str) -> list[float]:
    """Embed one query, retrying transient provider failures."""
    (vector,) = await embed([query], input_type=EmbedInput.QUERY)
    return vector


async def _tune_hnsw_walk(session: AsyncSession, candidates: int) -> None:
    """Let the HNSW walk resume past filtered and dead tuples until the candidate pool is met.

    strict_order because fusion scores rank position, which an approximate order makes
    meaningless; both settings are transaction-local, so neither leaks across the pool.
    """
    ef_search = min(candidates * config.EF_SEARCH_PER_CANDIDATE, EF_SEARCH_MAX)
    stmt = text(
        "SELECT set_config('hnsw.iterative_scan', 'strict_order', true),"
        " set_config('hnsw.ef_search', :ef_search, true)"
    ).bindparams(ef_search=str(ef_search))
    await session.execute(stmt)


def _filtered(stmt: Select, filters: SearchFilters) -> Select:
    """Narrow a candidate query before its limit, so the pool the fusion sees is honest."""
    if filters.celex is not None:
        stmt = stmt.where(DocumentChunk.celex == filters.celex)
    if filters.topic is not None:
        stmt = stmt.where(DocumentChunk.topic == filters.topic)
    return stmt


def _ranked(stmt: Select, order: Sequence, filters: SearchFilters, limit: int) -> Select:
    """A leg's top chunk ids, each carrying its 1-based position in that leg."""
    return (
        _filtered(stmt, filters)
        .add_columns(func.row_number().over(order_by=order).label("rank"))
        .order_by(*order)
        .limit(limit)
    )


def _vector_candidates(embedding: Sequence[float], filters: SearchFilters, limit: int) -> Select:
    """Chunk ids nearest the query vector by cosine distance, closest first."""
    order = (DocumentChunk.embedding.cosine_distance(embedding), DocumentChunk.id)
    stmt = select(DocumentChunk.id).where(DocumentChunk.embedding.is_not(None))
    return _ranked(stmt, order, filters, limit)


def _text_candidates(query: str, filters: SearchFilters, limit: int) -> Select:
    """Chunk ids whose search vector matches the query, best-ranked first."""
    tsquery = func.websearch_to_tsquery("english", query)
    order = (func.ts_rank_cd(DocumentChunk.search_vector, tsquery).desc(), DocumentChunk.id)
    stmt = select(DocumentChunk.id).where(DocumentChunk.search_vector.bool_op("@@")(tsquery))
    return _ranked(stmt, order, filters, limit)


async def hybrid_search(
    session: AsyncSession,
    *,
    query: str,
    embedding: Sequence[float],
    filters: SearchFilters,
    limit: int,
    candidates: int,
    rrf_k: int,
) -> tuple[SearchResult, ...]:
    """The corpus's best answers, both legs fused by 1/(rrf_k + rank) in one round trip.

    A full outer join keeps chunks only one leg found, whose missing term contributes nothing.
    """
    await _tune_hnsw_walk(session, candidates)
    by_vector = _vector_candidates(embedding, filters, candidates).cte("by_vector")
    by_text = _text_candidates(query, filters, candidates).cte("by_text")
    score = (
        func.coalesce(1.0 / (rrf_k + by_vector.c.rank), 0.0)
        + func.coalesce(1.0 / (rrf_k + by_text.c.rank), 0.0)
    ).label("score")
    joined = by_vector.join(by_text, by_vector.c.id == by_text.c.id, full=True).join(
        DocumentChunk, DocumentChunk.id == func.coalesce(by_vector.c.id, by_text.c.id)
    )
    stmt = (
        select(
            *CHUNK_COLUMNS,
            score,
            by_vector.c.rank.label("vector_rank"),
            by_text.c.rank.label("text_rank"),
        )
        .select_from(joined)
        .order_by(score.desc(), DocumentChunk.id)
        .limit(limit)
    )
    rows = await session.execute(stmt)
    return tuple(SearchResult.model_validate(row) for row in rows)


async def search(session: AsyncSession, request: SearchRequest) -> tuple[SearchResult, ...]:
    """The corpus's best answers to a query, fused across both legs and reranked."""
    limit = request.limit or config.SEARCH_DEFAULT_LIMIT
    embedding = await _embed_query(request.query)
    pool = max(limit, config.RERANK_POOL) if config.RERANK_ENABLED else limit
    results = await hybrid_search(
        session,
        query=request.query,
        embedding=embedding,
        filters=request.filters,
        limit=pool,
        candidates=config.SEARCH_CANDIDATES,
        rrf_k=config.RRF_K,
    )
    if config.RERANK_ENABLED:
        results = await rerank_results(request.query, results, limit=limit)
    return results
