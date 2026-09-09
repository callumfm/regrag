"""Voyage embeddings through LiteLLM: one call, wrapped errors."""

import logging
from enum import StrEnum
from operator import itemgetter

import litellm

from app.core.config import EMBED_DIMENSIONS, config
from app.core.llm.errors import LLMError, wrap_provider_errors

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 128
"""Voyage's ceiling on texts per embedding request."""


class EmbedInput(StrEnum):
    """Which side of an asymmetric embedding a text is on."""

    DOCUMENT = "document"
    QUERY = "query"


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
