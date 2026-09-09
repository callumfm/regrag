"""Model provider calls through LiteLLM: the error contract every call shares, what a call
spends, the embedding call, and the cache that replays a call from disk."""

import litellm

litellm.suppress_debug_info = True
"""A failure is logged as ours; litellm's help banner has no place on it."""
