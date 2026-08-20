"""Replay a run's paid retrieval calls from disk, so only the first run pays for them."""

import litellm
from litellm.caching import Cache
from litellm.types.caching import CachingSupportedCallTypes, LiteLLMCacheType

from app.core.config import config

CACHED_CALLS: list[CachingSupportedCallTypes] = ["aembedding", "arerank"]
"""The paid calls retrieval makes. Synthesis is left out on purpose: a run replaying its
own answers would measure the cache rather than the model."""


def enable_call_cache() -> None:
    """Serve repeated embed and rerank calls from EVAL_CACHE_DIR.

    litellm keys each call on its own request parameters — the rerank on its model, query
    and documents in order — so a re-ingest or a model change lands on a different key and
    nothing goes stale. Deleting the directory is the whole invalidation story.
    """
    litellm.cache = Cache(
        type=LiteLLMCacheType.DISK,
        disk_cache_dir=str(config.EVAL_CACHE_DIR),
        supported_call_types=CACHED_CALLS,
    )
