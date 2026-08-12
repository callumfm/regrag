"""Cross-encoder rerank of fused results: one Voyage call, degrading to the fused order."""

import logging

import litellm
from litellm.types.rerank import RerankResponseResult

from app.core.config import config
from app.core.llm import TRANSIENT_PROVIDER_ERRORS, LLMError, ProviderError, llm_retry
from app.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


@llm_retry
async def _rerank(query: str, documents: list[str]) -> list[RerankResponseResult]:
    """The cross-encoder's ranking of the documents, which the provider returns best-first."""
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
    return response.results or []


async def rerank_results(
    query: str, results: tuple[SearchResult, ...], *, limit: int
) -> tuple[SearchResult, ...]:
    """Results reordered by the cross-encoder and cut, or the fused order when rerank fails."""
    if not results:
        return ()
    try:
        ranked = await _rerank(query, [result.text for result in results])
    except LLMError:
        logger.warning("rerank failed, keeping the fused order")
        return results[:limit]
    return tuple(results[item["index"]] for item in ranked[:limit])
