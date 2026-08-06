"""Voyage embeddings through LiteLLM: one call, wrapped errors."""

import logging
import os
from collections.abc import Sequence
from enum import StrEnum

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

import litellm
from fastapi import status
from openai import OpenAIError as ProviderError

from app.core.config import config
from app.core.exceptions import DomainError

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


class EmbedInput(StrEnum):
    """Which side of an asymmetric embedding a text is on."""

    DOCUMENT = "document"
    QUERY = "query"


class LLMError(DomainError):
    """A model provider call failed after its retries were exhausted."""

    status_code = status.HTTP_502_BAD_GATEWAY


async def embed(texts: Sequence[str], *, input_type: EmbedInput) -> list[list[float]]:
    """Embed texts in one provider call, preserving input order."""
    if not texts:
        return []
    try:
        response = await litellm.aembedding(
            model=config.EMBED_MODEL,
            input=list(texts),
            input_type=input_type.value,
            dimensions=config.EMBED_DIMENSIONS,
            api_key=config.VOYAGE_API_KEY,
            timeout=config.EMBED_TIMEOUT,
            num_retries=config.EMBED_MAX_RETRIES,
        )
    except ProviderError as exc:
        logger.warning("embedding call failed: %s", exc)
        raise LLMError("embedding call failed") from exc
    return [item["embedding"] for item in response.data]
