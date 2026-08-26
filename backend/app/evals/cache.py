"""Replay a run's paid retrieval calls from disk, so only the first run pays for them."""

import litellm
from litellm.caching import Cache
from litellm.types.caching import CachingSupportedCallTypes, LiteLLMCacheType

from app.core.config import config

CACHED_CALLS: list[CachingSupportedCallTypes] = ["aembedding", "arerank"]
"""The paid calls retrieval makes. Synthesis is left out on purpose: a run replaying its
own answers would measure the cache rather than the model."""


def enable_call_cache() -> None:
    """Serve repeated embed and rerank calls from EVAL_CACHE_DIR, keyed on each call's own
    request parameters (see the key tests). Deleting the directory invalidates the lot."""
    litellm.enable_caching_on_provider_specific_optional_params = True
    litellm.cache = Cache(
        type=LiteLLMCacheType.DISK,
        disk_cache_dir=str(config.EVAL_CACHE_DIR),
        supported_call_types=CACHED_CALLS,
    )
