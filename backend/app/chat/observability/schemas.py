"""Chat request tracking: one row per handled question, the ledger a spend cap sums over."""

from sqlalchemy import Index
from sqlalchemy.orm import Mapped, mapped_column

from app.chat.enums import ChatOutcome
from app.core.db.schema import BaseSchema


class ChatRequest(BaseSchema):
    """One handled question: its outcome, timings, source count and token usage.
    A spend cap sums tokens over a recent window; the index keeps that off a full scan."""

    __tablename__ = "chat_requests"
    __table_args__ = (Index("ix_chat_requests_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str | None]
    question: Mapped[str]
    outcome: Mapped[ChatOutcome]
    model: Mapped[str]
    retrieve_ms: Mapped[int | None]
    ttft_ms: Mapped[int | None]
    total_ms: Mapped[int]
    sources: Mapped[int]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
