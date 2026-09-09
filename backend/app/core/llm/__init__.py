"""Model provider calls through LiteLLM: the error contract every call shares, what a call
spends, and the embedding call. LiteLLM's process settings are applied here, before any
submodule imports it, so its cost map is read from disk rather than fetched."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

import litellm  # noqa: E402

litellm.suppress_debug_info = True
