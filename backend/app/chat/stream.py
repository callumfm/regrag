"""Chat streaming: the graph run, translated into chat events, and recorded when it ends."""

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import anyio
from sqlalchemy.exc import SQLAlchemyError

from app.chat.enums import ChatNode
from app.chat.graph import chat_graph
from app.chat.models import (
    ChatEvent,
    ChatState,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    TextEvent,
)
from app.chat.service import create_chat_request
from app.core.clock import elapsed_ms
from app.core.db.session import get_session
from app.core.exceptions import DomainError, describe
from app.core.logger import request_id_var
from app.core.models import ErrorResponse

logger = logging.getLogger(__name__)


def _error_name(exc: Exception) -> str:
    """What the ledger records a failed run as: a DomainError's message, or the type of an
    unexpected one — the wire message says only that something went wrong."""
    return exc.message if isinstance(exc, DomainError) else type(exc).__name__


def _error_event(exc: Exception) -> ErrorEvent:
    """The error event for a failed stream, logged like the JSON handlers log theirs."""
    error, message = describe(exc)
    if isinstance(exc, DomainError):
        logger.warning("chat stream failed: %s", message)
    else:
        logger.exception("chat stream failed unexpectedly")
    body = ErrorResponse(error=error, message=message, request_id=request_id_var.get())
    return ErrorEvent(data=body)


async def _stream_graph_events(state: ChatState) -> AsyncGenerator[ChatEvent, None]:
    """The graph run as (mode, payload) pairs, read as events: sources once retrieve returns,
    the refusal once refuse does, each model chunk's text; then done. Each values snapshot
    is folded onto the one state, so it holds the run at the end."""
    stream: AsyncIterator[Any] = chat_graph.astream(state, stream_mode=["values", "messages"])
    async for mode, item in stream:
        if mode == "values":
            state.refresh(item)
            if state.last_node is ChatNode.RETRIEVE:
                yield SourcesEvent.from_results(state.sources)
            elif state.last_node is ChatNode.REFUSE:
                yield TextEvent(data=state.answer)
        else:
            chunk, _metadata = item
            if text := chunk.text:
                yield TextEvent(data=text)
    yield DoneEvent()


async def stream_chat_events(question: str) -> AsyncGenerator[ChatEvent, None]:
    """One question's events, ended by an error event if the run raises; however it ends —
    done, refused, error, or the client leaving, which cancels this task — it is recorded as
    one chat request, in its own session, shielded from that cancellation. A failed write is
    logged, not raised: the answer already went out."""
    state = ChatState(question=question)
    start = time.perf_counter()
    try:
        async for event in _stream_graph_events(state):
            yield event
    except Exception as exc:
        state.error = _error_name(exc)
        yield _error_event(exc)
    finally:
        state.total_ms = elapsed_ms(start)
        with anyio.CancelScope(shield=True):
            try:
                async with get_session(auto_commit=False) as session:
                    await create_chat_request(session, state)
            except SQLAlchemyError:
                logger.exception("chat request not recorded")
