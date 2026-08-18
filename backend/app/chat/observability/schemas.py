"""Chat run tracking: one row per streamed answer, the ledger a spend cap sums over."""

from sqlalchemy import Index
from sqlalchemy.orm import Mapped, mapped_column

from app.chat.observability.enums import ChatOutcome
from app.core.db.schema import BaseSchema


class ChatRun(BaseSchema):
    __tablename__ = "chat_runs"
    __table_args__ = (Index("ix_chat_runs_created_at", "created_at"),)
    """A spend cap sums tokens over a recent window; the index keeps that off a full scan."""

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str | None]
    outcome: Mapped[ChatOutcome]
    model: Mapped[str]
    retrieve_ms: Mapped[int | None]
    """NULL when the run ended before retrieval did; likewise ttft_ms before the first token."""
    ttft_ms: Mapped[int | None]
    total_ms: Mapped[int]
    sources: Mapped[int]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    """NULL when no usage came back: the model call never finished, or the provider sent none."""
