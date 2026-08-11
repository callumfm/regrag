"""Voyage embeddings through LiteLLM: one call, wrapped errors."""

import logging
import os
from enum import StrEnum
from operator import itemgetter

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

import litellm
from fastapi import status
from litellm.exceptions import ServiceUnavailableError
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from openai import (
    OpenAIError as ProviderError,
)

from app.core.config import EMBED_DIMENSIONS, config
from app.core.exceptions import DomainError
from app.core.retry import transient_retry

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 128
"""Voyage's ceiling on texts per embedding request."""

TRANSIENT_PROVIDER_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    ServiceUnavailableError,
)
"""Provider failures worth retrying; 400, 401, 403 and 404 never are."""


class EmbedInput(StrEnum):
    """Which side of an asymmetric embedding a text is on."""

    DOCUMENT = "document"
    QUERY = "query"


class LLMError(DomainError):
    """A model provider call failed, or returned a response we cannot trust."""

    status_code = status.HTTP_502_BAD_GATEWAY

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def _is_transient(exc: BaseException) -> bool:
    """Provider failures the wrap point judged worth another attempt."""
    return isinstance(exc, LLMError) and exc.transient


llm_retry = transient_retry(_is_transient)
"""Decorator retrying transient provider failures with exponential backoff."""


async def embed(texts: list[str], *, input_type: EmbedInput) -> list[list[float]]:
    """Embed texts in one provider call, in input order. Retries are the caller's."""
    if not texts:
        return []
    try:
        response = await litellm.aembedding(
            model=config.EMBED_MODEL,
            input=texts,
            input_type=input_type.value,
            dimensions=EMBED_DIMENSIONS,
            api_key=config.VOYAGE_API_KEY,
            timeout=config.EMBED_TIMEOUT,
        )
    except ProviderError as exc:
        logger.warning("embedding call failed: %s", exc)
        raise LLMError(
            "embedding call failed", transient=isinstance(exc, TRANSIENT_PROVIDER_ERRORS)
        ) from exc
    if len(response.data) != len(texts):
        logger.warning(
            "embedding response misaligned: got %d items for %d inputs",
            len(response.data),
            len(texts),
        )
        raise LLMError("embedding call failed")
    return [item["embedding"] for item in sorted(response.data, key=itemgetter("index"))]
