"""What tune may vary, and the values worth trying by default."""

from typing import Any

from app.core.config import Config

TUNABLE_PARAMS: dict[str, tuple[Any, ...]] = {
    "CHAT_SOURCES": (3, 5, 8),
    "CHAT_CONTEXT_CHUNKS": (10, 15, 20, 30),
    "EXPAND_SECTIONS": (True, False),
    "RERANK_ENABLED": (True, False),
    "MIN_COSINE_SIMILARITY": (0.20, 0.30, 0.40),
    "MIN_RERANKER_RELEVANCE": (0.35, 0.45, 0.55),
}


def validate_value(name: str, value: Any) -> Any:
    """The value as the config would hold it, coerced and bounded by its own field."""
    if name not in TUNABLE_PARAMS:
        raise ValueError(
            f"{name} is not a tunable param; expected one of: {', '.join(TUNABLE_PARAMS)}"
        )
    if name not in Config.model_fields:
        raise ValueError(f"{name} is no longer a config field; update TUNABLE_PARAMS")
    return getattr(Config.model_validate({name: value}), name)


def get_tunable_params() -> dict[str, tuple[Any, ...]]:
    """The curated params, every name and default value revalidated against the live
    config, so a renamed or retightened setting fails here rather than mid-sweep."""
    return {
        name: tuple(validate_value(name, value) for value in values)
        for name, values in TUNABLE_PARAMS.items()
    }
