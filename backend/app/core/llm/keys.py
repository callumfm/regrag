"""Which configured key a model call is made with."""

from litellm import get_llm_provider

from app.core.config import config


def api_key_for(model: str) -> str:
    """The configured key for the provider a model names, so any role can be pointed at
    any provider that has a setting; empty when none is set, and the provider refuses."""
    provider = get_llm_provider(model)[1]
    return getattr(config, f"{provider.upper()}_API_KEY").get_secret_value()
