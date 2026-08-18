"""Chat request rows: the stream's stats logged and persisted once it has ended."""

import logging

from sqlalchemy.exc import SQLAlchemyError

from app.chat.enums import ChatOutcome
from app.chat.observability.models import RequestStats
from app.chat.observability.schemas import ChatRequest
from app.core.clock import elapsed_ms
from app.core.config import config
from app.core.db.crud import create_record
from app.core.db.session import get_session
from app.core.logger import request_id_var

logger = logging.getLogger(__name__)


def log_request(request: ChatRequest) -> None:
    """The one stats line per stream; the request ID rides in on the log context."""
    fields = {
        "outcome": request.outcome,
        "retrieve_ms": request.retrieve_ms,
        "ttft_ms": request.ttft_ms,
        "total_ms": request.total_ms,
        "sources": request.sources,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
    }
    logger.info(
        "chat %(outcome)s - retrieve %(retrieve_ms)sms, first token %(ttft_ms)sms, "
        "total %(total_ms)sms, %(sources)s sources, %(input_tokens)s/%(output_tokens)s tokens",
        fields,
        extra=fields,
    )


async def record_request(question: str, stats: RequestStats, outcome: ChatOutcome) -> None:
    """Log the stream's stats and persist them as a chat_requests row.

    Runs once the stream has ended, outside any request scope, so it owns its session —
    and a failed write is logged, not raised: the answer already went out and
    observability must not turn it into an error.
    """
    usage = stats.usage
    request = ChatRequest(
        request_id=request_id_var.get(),
        question=question,
        outcome=outcome,
        model=config.CHAT_MODEL,
        retrieve_ms=stats.retrieve_ms,
        ttft_ms=stats.ttft_ms,
        total_ms=elapsed_ms(stats.start),
        sources=stats.sources,
        input_tokens=usage["input_tokens"] if usage else None,
        output_tokens=usage["output_tokens"] if usage else None,
    )
    log_request(request)
    try:
        async with get_session(auto_commit=False) as session:
            await create_record(session, request)
    except SQLAlchemyError:
        logger.exception("chat request not recorded")
