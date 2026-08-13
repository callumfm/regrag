"""Chat test fakes shared across the chat test modules."""

from typing import Any

from app.retrieval.models import SearchResult


def make_result(**overrides: Any) -> SearchResult:
    """A retrieved chunk with sane defaults, overridable per field."""
    defaults: dict[str, Any] = {
        "id": 1,
        "celex": "32023R1805",
        "topic": "fueleu",
        "citation": "Article 4(1)",
        "title": "Greenhouse gas intensity limit",
        "text": "The greenhouse gas intensity of the energy used on board.",
        "score": 0.9,
        "vector_rank": 1,
        "text_rank": 1,
    }
    return SearchResult(**{**defaults, **overrides})
