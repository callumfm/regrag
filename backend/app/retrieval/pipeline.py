"""Search: embed the query, then let one query run both legs and fuse their ranks."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import EmbedInput, embed, llm_retry
from app.retrieval.constants import CANDIDATES, DEFAULT_LIMIT, RRF_K
from app.retrieval.models import SearchFilters, SearchResult
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
    limit: int = DEFAULT_LIMIT,
    candidates: int = CANDIDATES,
    k: int = RRF_K,
) -> tuple[SearchResult, ...]:
    """The corpus's best answers to a query, fused across the vector and keyword legs."""
    embedding = await _embed_query(query)
    return await hybrid_search(
        session,
        embedding,
        query,
        filters or SearchFilters(),
        candidates=candidates,
        k=k,
        limit=limit,
    )
