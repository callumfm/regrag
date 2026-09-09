"""Replay repeated provider calls from disk, so only the first run pays for them."""

from pathlib import Path

import litellm
from litellm.caching import Cache
from litellm.types.caching import LiteLLMCacheType


def enable_call_cache(directory: Path) -> None:
    """Serve repeated embed and rerank calls from the directory, keyed on each call's own
    request parameters, a provider's own included (see the key tests). A completion is
    never replayed: served from disk it would measure the cache rather than the model.
    Deleting the directory invalidates the lot."""
    litellm.enable_caching_on_provider_specific_optional_params = True
    litellm.cache = Cache(
        type=LiteLLMCacheType.DISK,
        disk_cache_dir=str(directory),
        supported_call_types=["aembedding", "arerank"],
    )


def call_cache_enabled() -> bool:
    """Whether a call cache is installed, read off litellm itself rather than a caller's
    word, so what a run records cannot disagree with what served its calls."""
    return litellm.cache is not None
