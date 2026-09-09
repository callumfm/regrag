"""RegRag. LiteLLM's cost map is pinned to the copy it ships before any module imports it,
so no process makes a network call at import time."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

__version__ = "0.1.0"
