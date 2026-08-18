"""Minimum retrieval score for sources to prevent hallucinations and unnecessary llm calls."""

from collections.abc import Sequence

from app.core.config import config
from app.retrieval.models import SearchResult


def meets_thresholds(hits: Sequence[SearchResult]) -> bool:
    """Assert that the retrieved documents meet a minimum score threshold for answer generation."""
    if not hits:
        return False

    thresholds = (
        ("cosine_similarity", config.CHAT_MIN_COSINE_SIMILARITY),
        ("reranker_relevance", config.CHAT_MIN_RERANKER_RELEVANCE),
    )
    for attribute, threshold in thresholds:
        values = [value for hit in hits if (value := getattr(hit, attribute)) is not None]

        if values and max(values) < threshold:
            return False

    return True
