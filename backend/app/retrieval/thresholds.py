"""Minimum retrieval score for sources, to prevent hallucinations and wasted model calls."""

from collections.abc import Sequence

from app.core.config import config
from app.retrieval.models import SearchResult


def meets_thresholds(hits: Sequence[SearchResult]) -> bool:
    """Whether the retrieved documents clear every score bar they carry a signal for."""
    if not hits:
        return False

    cosines = [hit.cosine_similarity for hit in hits if hit.cosine_similarity is not None]
    if cosines and max(cosines) < config.MIN_COSINE_SIMILARITY:
        return False

    relevances = [hit.reranker_relevance for hit in hits if hit.reranker_relevance is not None]
    if relevances and max(relevances) < config.MIN_RERANKER_RELEVANCE:
        return False

    return True
