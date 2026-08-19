"""Chat stream orchestration: the graph run, translated into chat events, logged and recorded."""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

import anyio
from sqlalchemy.exc import SQLAlchemyError

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
from app.chat.schemas import ChatRequest
from app.core.clock import elapsed_ms
from app.core.config import config
from app.core.db.crud import create_record
from app.core.db.session import get_session
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


def log_request(request: ChatRequest) -> None:
    """The one stats line per stream; the request ID rides in on the log context."""
    fields = {
        "outcome": request.outcome,
        "nodes": ">".join(request.nodes),
        "retrieve_ms": request.retrieve_ms,
        "ttft_ms": request.ttft_ms,
        "total_ms": request.total_ms,
        "sources": request.sources,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
    }
    logger.info(
        "chat %(outcome)s [%(nodes)s] - retrieve %(retrieve_ms)sms, first token %(ttft_ms)sms, "
        "total %(total_ms)sms, %(sources)s sources, %(input_tokens)s/%(output_tokens)s tokens",
        fields,
        extra=fields,
    )


async def record_request(
    state: ChatState, outcome: ChatOutcome, ttft_ms: int | None, total_ms: int
) -> None:
    """Log what the run produced and persist it as a chat_requests row.

    Runs once the stream has ended, outside any request scope, so it owns its session —
    and a failed write is logged, not raised: the answer already went out and
    observability must not turn it into an error.
    """
    usage = state.usage
    request = ChatRequest(
        request_id=request_id_var.get(),
        question=state.question,
        outcome=outcome,
        nodes=[node.value for node in state.nodes],
        model=config.CHAT_MODEL,
        retrieve_ms=state.retrieve_ms,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        sources=len(state.sources),
        input_tokens=usage["input_tokens"] if usage else None,
        output_tokens=usage["output_tokens"] if usage else None,
    )
    log_request(request)
    try:
        async with get_session(auto_commit=False) as session:
            await create_record(session, request)
    except SQLAlchemyError:
        logger.exception("chat request not recorded")


async def chat_events(question: str) -> AsyncGenerator[ChatEvent, None]:
    """Sources once, then tokens, then done; an error event ends a failed stream. A refused
    question sends its refusal as the one token, after an empty sources event.
    However it ends — done, refused, error, or the client leaving, which cancels this
    task — one chat request is recorded, so the write is shielded from that cancellation."""
    start = time.perf_counter()
    ttft_ms: int | None = None
    outcome = ChatOutcome.ABORTED
    state = ChatState(question=question)
    announced: set[ChatNode] = set()
    stream = chat_graph.astream(state, stream_mode=["values", "messages"])
    try:
        async for mode, item in stream:
            if mode == "values":
                state = ChatState(**cast(dict[str, Any], item))
                for node in (n for n in state.nodes if n not in announced):
                    announced.add(node)
                    if node is ChatNode.RETRIEVE:
                        yield SourcesEvent.from_results(state.sources)
                    elif node is ChatNode.REFUSE:
                        ttft_ms = elapsed_ms(start)
                        yield TokenEvent(data=ChatToken(text=state.answer))
            elif text := cast(tuple[Any, Any], item)[0].text:
                ttft_ms = elapsed_ms(start) if ttft_ms is None else ttft_ms
                yield TokenEvent(data=ChatToken(text=text))

        yield DoneEvent()
        outcome = ChatOutcome.REFUSED if ChatNode.REFUSE in state.nodes else ChatOutcome.DONE
    except Exception as exc:
        outcome = ChatOutcome.ERROR
        yield _error_event(exc)
    finally:
        with anyio.CancelScope(shield=True):
            await record_request(state, outcome, ttft_ms, elapsed_ms(start))
