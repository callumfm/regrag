"""SSE events: how each item of the chat graph's stream reaches the client."""

import logging
from typing import Any

from fastapi.sse import ServerSentEvent

from app.chat.models import ChatSource, ChatToken
from app.core.exceptions import DomainError, describe
from app.core.logger import request_id_var
from app.core.models import ErrorResponse
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def sse_event(name: str, payload: Any) -> ServerSentEvent:
    """One SSE event carrying a JSON payload."""
    return ServerSentEvent(event=name, data=payload)


def sources_event(sources: tuple[RetrievedChunk, ...]) -> ServerSentEvent:
    """The sources event binding [n] markers to the retrieved chunks."""
    payload = [
        ChatSource.from_result(marker, result).model_dump()
        for marker, result in enumerate(sources, start=1)
    ]
    return sse_event("sources", payload)


def error_event(exc: Exception) -> ServerSentEvent:
    """The error event in the app's one error shape, logged like the JSON handlers."""
    error, message = describe(exc)
    if isinstance(exc, DomainError):
        logger.warning("chat stream failed: %s", message)
    else:
        logger.exception("chat stream failed unexpectedly")
    body = ErrorResponse(error=error, message=message, request_id=request_id_var.get())
    return sse_event("error", body.model_dump(exclude_none=True))


def event_for(mode: str, data: Any) -> ServerSentEvent | None:
    """The SSE event one stream item maps to, if any: sources from the retrieve
    update, a token from each message chunk carrying text."""
    if mode == "updates" and "retrieve" in data:
        return sources_event(data["retrieve"]["sources"])
    if mode == "messages":
        chunk, _ = data
        if text := chunk.text:
            return sse_event("token", ChatToken(text=text).model_dump())
    return None
