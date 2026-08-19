"""Voyage embeddings through LiteLLM: one call, wrapped errors."""

import functools
import logging
import os
from collections.abc import Awaitable, Callable, Coroutine
from enum import StrEnum
from operator import itemgetter
from typing import Any

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


def wrap_provider_errors[**P, R](
    label: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Translate the wrapped call's provider failure into an LLMError reading
    "<label> failed", with the provider's own text kept to the log."""

    def decorate(fn: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return await fn(*args, **kwargs)
            except ProviderError as exc:
                logger.warning("%s failed: %s", label, exc)
                raise LLMError(
                    f"{label} failed", transient=isinstance(exc, TRANSIENT_PROVIDER_ERRORS)
                ) from exc

        return wrapper

    return decorate


@wrap_provider_errors("embedding call")
async def embed(texts: list[str], *, input_type: EmbedInput) -> list[list[float]]:
    """Embed texts in one provider call, in input order. Retries are the caller's."""
    if not texts:
        return []
    response = await litellm.aembedding(
        model=config.EMBED_MODEL,
        input=texts,
        input_type=input_type.value,
        dimensions=EMBED_DIMENSIONS,
        api_key=config.VOYAGE_API_KEY.get_secret_value(),
        timeout=config.EMBED_TIMEOUT,
    )
    if len(response.data) != len(texts):
        logger.warning(
            "embedding response misaligned: got %d items for %d inputs",
            len(response.data),
            len(texts),
        )
        raise LLMError("embedding call failed")
    return [item["embedding"] for item in sorted(response.data, key=itemgetter("index"))]
