"""Chat request recording: one row per handled question, with a row per step it ran through."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatState
from app.chat.schemas import ChatRequest, ChatRequestStep
from app.core.config import config
from app.core.db.crud import create_record
from app.core.logger import request_id_var

logger = logging.getLogger(__name__)


async def create_chat_request(session: AsyncSession, state: ChatState) -> None:
    """The run as recorded: one stats line, and a chat_requests row with a row per step."""
    fields = state.log_fields()
    logger.info("chat %(outcome)s in %(total_ms)sms", fields, extra=fields)
    input_tokens, output_tokens = state.token_totals()
    steps = [
        ChatRequestStep(
            position=idx,
            step=result.step.value,
            ms=result.ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        for idx, result in enumerate(state.steps)
    ]
    request = ChatRequest(
        request_id=request_id_var.get(),
        question=state.question,
        outcome=state.outcome,
        model=config.CHAT_MODEL,
        total_ms=state.total_ms,
        sources=len(state.sources),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=state.error,
        steps=steps,
    )
    await create_record(session, request)
