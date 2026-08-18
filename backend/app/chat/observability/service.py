"""Chat run rows: the stream's stats logged and persisted once it has ended."""

import logging

from app.chat.enums import ChatOutcome
from app.chat.observability.models import StreamStats
from app.chat.observability.schemas import ChatRun
from app.core.config import config
from app.core.db.crud import create_record
from app.core.db.session import get_session
from app.core.logger import request_id_var

logger = logging.getLogger(__name__)


def chat_run(stats: StreamStats, outcome: ChatOutcome) -> ChatRun:
    """The row one finished stream leaves behind."""
    usage = stats.usage
    return ChatRun(
        request_id=request_id_var.get(),
        outcome=outcome,
        model=config.CHAT_MODEL,
        retrieve_ms=stats.retrieve_ms,
        ttft_ms=stats.ttft_ms,
        total_ms=stats.elapsed_ms(),
        sources=stats.sources,
        input_tokens=usage["input_tokens"] if usage else None,
        output_tokens=usage["output_tokens"] if usage else None,
    )


def log_run(run: ChatRun) -> None:
    """The one stats line per stream; the request ID rides in on the log context."""
    logger.info(
        "chat %s - retrieve %sms, first token %sms, total %sms, %s sources, %s/%s tokens",
        run.outcome,
        run.retrieve_ms,
        run.ttft_ms,
        run.total_ms,
        run.sources,
        run.input_tokens,
        run.output_tokens,
        extra={
            "outcome": run.outcome,
            "retrieve_ms": run.retrieve_ms,
            "ttft_ms": run.ttft_ms,
            "total_ms": run.total_ms,
            "sources": run.sources,
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
        },
    )


async def record_run(stats: StreamStats, outcome: ChatOutcome) -> None:
    """Log the stream's stats and persist them as a chat_runs row.

    Runs after the client has its last event, so a failed write is logged, not raised:
    the answer already went out and observability must not turn it into an error.
    """
    run = chat_run(stats, outcome)
    log_run(run)
    try:
        async with get_session(auto_commit=False) as session:
            await create_record(session, run)
    except Exception:
        logger.exception("chat run not recorded")
