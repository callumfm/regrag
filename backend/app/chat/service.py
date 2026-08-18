"""Chat stream orchestration: the graph run, translated into chat events and measured."""

import logging
from collections.abc import AsyncGenerator
from typing import Any, cast

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
from app.chat.observability.models import StreamStats
from app.chat.observability.service import record_run
from app.core.exceptions import DomainError, describe
from app.core.logger import request_id_var
from app.core.models import ErrorResponse

logger = logging.getLogger(__name__)


def error_event(exc: Exception) -> ErrorEvent:
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
    However it ends — done, error, or the client leaving — one chat run is recorded."""
    stats = StreamStats()
    outcome = ChatOutcome.ABORTED
    try:
        async for item in chat_graph.astream(
            ChatState(question=question), stream_mode=["updates", "messages"]
        ):
            mode, data = cast(tuple[str, Any], item)
            if mode == "updates" and ChatNode.RETRIEVE in data:
                sources = data[ChatNode.RETRIEVE]["sources"]
                stats.retrieve_ms, stats.sources = stats.elapsed_ms(), len(sources)
                yield SourcesEvent.from_results(sources)
            elif mode == "updates" and ChatNode.SYNTHESIZE in data:
                stats.usage = data[ChatNode.SYNTHESIZE]["usage"]
            elif mode == "messages" and (text := data[0].text):
                if stats.ttft_ms is None:
                    stats.ttft_ms = stats.elapsed_ms()
                yield TokenEvent(data=ChatToken(text=text))
        yield DoneEvent()
        outcome = ChatOutcome.DONE
    except Exception as exc:
        outcome = ChatOutcome.ERROR
        yield error_event(exc)
    finally:
        await record_run(stats, outcome)
