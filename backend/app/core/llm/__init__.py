"""Model provider calls through LiteLLM: which key a call is made with, the error contract
every call shares, what a call spends, the embedding call, and the cache that replays a
call from disk."""

import litellm

litellm.suppress_debug_info = True
"""A failure is logged as ours; litellm's help banner has no place on it."""

litellm.drop_params = True
"""A parameter the model does not accept is dropped rather than branched on, so pointing a
role at another provider is a setting rather than a per-model special case."""
