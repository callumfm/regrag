"""Cross-encoder rerank of fused results: one Voyage call, degrading to the fused order."""

import logging
from operator import attrgetter

import litellm

from app.core.config import config
from app.core.llm import TRANSIENT_PROVIDER_ERRORS, LLMError, ProviderError, llm_retry
from app.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


async def _rerank(query: str, documents: list[str]) -> list[int]:
    """Document indices best-first, per the cross-encoder. Retries are the caller's."""
    try:
        response = await litellm.arerank(
            model=config.RERANK_MODEL,
            query=query,
            documents=documents,
            api_key=config.VOYAGE_API_KEY,
            timeout=config.RERANK_TIMEOUT,
        )
    except ProviderError as exc:
        logger.warning("rerank call failed: %s", exc)
        raise LLMError(
            "rerank call failed", transient=isinstance(exc, TRANSIENT_PROVIDER_ERRORS)
        ) from exc
    ranked = sorted(response.results or [], key=attrgetter("relevance_score"), reverse=True)
    order = [result.index for result in ranked]
    if sorted(order) != list(range(len(documents))):
        logger.warning(
            "rerank response misaligned: got %d items for %d documents",
            len(order),
            len(documents),
        )
        raise LLMError("rerank call failed")
    return order


_rerank_with_retry = llm_retry(_rerank)


async def rerank_results(
    query: str, results: tuple[SearchResult, ...], *, limit: int
) -> tuple[SearchResult, ...]:
    """Results reordered by the cross-encoder and cut, or the fused order when rerank fails."""
    if not results:
        return ()
    try:
        order = await _rerank_with_retry(query, [result.text for result in results])
    except LLMError:
        logger.warning("rerank failed, keeping the fused order")
        return results[:limit]
    return tuple(results[index] for index in order[:limit])
