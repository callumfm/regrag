"""Chat stream orchestration: graph output translated into SSE events."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.sse import ServerSentEvent

from app.chat.graph import ChatState, chat_graph
from app.chat.models import ChatSource, ChatToken
from app.core.exceptions import DomainError, describe
from app.core.logger import request_id_var
from app.core.models import ErrorResponse
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _event(name: str, payload: Any) -> ServerSentEvent:
    """One SSE event carrying a JSON payload."""
    return ServerSentEvent(event=name, data=payload)


def _sources_event(sources: tuple[RetrievedChunk, ...]) -> ServerSentEvent:
    """The sources event binding [n] markers to the retrieved chunks."""
    payload = [
        ChatSource.from_result(marker, result).model_dump()
        for marker, result in enumerate(sources, start=1)
    ]
    return _event("sources", payload)


def _error_event(exc: Exception) -> ServerSentEvent:
    """The error event in the app's one error shape, logged like the JSON handlers."""
    error, message = describe(exc)
    if isinstance(exc, DomainError):
        logger.warning("chat stream failed: %s", message)
    else:
        logger.exception("chat stream failed unexpectedly")
    body = ErrorResponse(error=error, message=message, request_id=request_id_var.get())
    return _event("error", body.model_dump(exclude_none=True))


def _event_for(mode: str, data: Any) -> ServerSentEvent | None:
    """The SSE event one stream item maps to, if any: sources from the retrieve
    update, a token from each message chunk carrying text."""
    if mode == "updates" and "retrieve" in data:
        return _sources_event(data["retrieve"]["sources"])
    if mode == "messages":
        chunk, _ = data
        if text := chunk.text:
            return _event("token", ChatToken(text=text).model_dump())
    return None


async def chat_events(question: str) -> AsyncGenerator[ServerSentEvent, None]:
    """Sources once, then tokens, then done; an error event ends a failed stream."""
    try:
        async for mode, data in chat_graph.astream(
            ChatState(question=question), stream_mode=["updates", "messages"]
        ):
            if event := _event_for(mode, data):
                yield event
    except Exception as exc:
        yield _error_event(exc)
        return
    yield _event("done", {})
