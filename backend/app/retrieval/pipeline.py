"""Search: embed the query, run both legs, fuse their ranks, load the winners."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import EmbedInput, embed, llm_retry
from app.retrieval.constants import CANDIDATES, DEFAULT_LIMIT, RRF_K
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import SearchFilters, SearchResult
from app.retrieval.service import hydrate, text_search, vector_search


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
    filters = filters or SearchFilters()
    embedding = await _embed_query(query)
    vector_ids = await vector_search(session, embedding, filters, limit=candidates)
    text_ids = await text_search(session, query, filters, limit=candidates)
    fused = reciprocal_rank_fusion(vector_ids, text_ids, k=k)[:limit]
    chunks = await hydrate(session, [rank.chunk_id for rank in fused])
    return tuple(
        SearchResult(
            **chunks[rank.chunk_id].model_dump(),
            score=rank.score,
            vector_rank=rank.vector_rank,
            text_rank=rank.text_rank,
        )
        for rank in fused
    )
