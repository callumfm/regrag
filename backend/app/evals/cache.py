"""Which calls an eval run replays from the call cache."""

from litellm.types.caching import CachingSupportedCallTypes

CACHED_CALLS: list[CachingSupportedCallTypes] = ["aembedding", "arerank"]
"""The paid calls retrieval makes. Synthesis is left out on purpose: a run replaying its
own answers would measure the cache rather than the model."""
