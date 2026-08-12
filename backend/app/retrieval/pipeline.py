"""Search: embed the query, let one query run both legs and fuse their ranks, then rerank."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.core.llm import EmbedInput, embed, llm_retry
from app.retrieval.models import SearchFilters, SearchResult
from app.retrieval.rerank import rerank_results
from app.retrieval.service import hybrid_search


@llm_retry
async def _embed_query(query: str) -> list[float]:
    """Embed one query, retrying transient provider failures."""
    (vector,) = await embed([query], input_type=EmbedInput.QUERY)
    return vector


async def search(
    session: AsyncSession,
    query: str,
    filters: SearchFilters | None = None,
    *,
    limit: int = config.SEARCH_DEFAULT_LIMIT,
    candidates: int = config.SEARCH_CANDIDATES,
    k: int = config.RRF_K,
) -> tuple[SearchResult, ...]:
    """The corpus's best answers to a query, fused across both legs and reranked."""
    embedding = await _embed_query(query)
    pool = max(limit, config.RERANK_POOL) if config.RERANK_ENABLED else limit
    results = await hybrid_search(
        session,
        embedding,
        query,
        filters or SearchFilters(),
        candidates=candidates,
        k=k,
        limit=pool,
    )
    if config.RERANK_ENABLED:
        results = await rerank_results(query, results, limit=limit)
    return results
