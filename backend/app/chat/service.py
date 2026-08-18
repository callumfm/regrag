"""Chat stream orchestration: the graph run, translated into chat events and measured."""

import logging
from collections.abc import AsyncGenerator
from typing import Any, cast

import anyio

from app.chat.enums import ChatNode, ChatOutcome
from app.chat.graph import chat_graph
from app.chat.models import (
    ChatEvent,
    ChatState,
    ChatToken,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    TokenEvent,
)
from app.chat.observability.models import RequestStats
from app.chat.observability.service import record_request
from app.core.exceptions import DomainError, describe
from app.core.logger import request_id_var
from app.core.models import ErrorResponse

logger = logging.getLogger(__name__)


def _error_event(exc: Exception) -> ErrorEvent:
    """The error event for a failed stream, logged like the JSON handlers log theirs."""
    error, message = describe(exc)
    if isinstance(exc, DomainError):
        logger.warning("chat stream failed: %s", message)
    else:
        logger.exception("chat stream failed unexpectedly")
    body = ErrorResponse(error=error, message=message, request_id=request_id_var.get())
    return ErrorEvent(data=body)


async def chat_events(question: str) -> AsyncGenerator[ChatEvent, None]:
    """Sources once, then tokens, then done; an error event ends a failed stream.
    However it ends — done, error, or the client leaving, which cancels this task —
    one chat request is recorded, so the write is shielded from that cancellation."""
    stats = RequestStats()
    outcome = ChatOutcome.ABORTED
    stream = chat_graph.astream(ChatState(question=question), stream_mode=["updates", "messages"])
    try:
        async for item in stream:
            match cast(tuple[str, Any], item):
                case ("updates", {ChatNode.RETRIEVE: {"sources": sources}}):
                    stats.retrieved(len(sources))
                    yield SourcesEvent.from_results(sources)

                case ("messages", (chunk, _)):
                    if chunk.usage_metadata:
                        stats.usage = chunk.usage_metadata
                    if text := chunk.text:
                        stats.token()
                        yield TokenEvent(data=ChatToken(text=text))

        yield DoneEvent()
        outcome = ChatOutcome.DONE
    except Exception as exc:
        outcome = ChatOutcome.ERROR
        yield _error_event(exc)
    finally:
        with anyio.CancelScope(shield=True):
            await record_request(question, stats, outcome)
