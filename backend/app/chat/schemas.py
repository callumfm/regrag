"""Chat request tracking: one row per handled question, the ledger a spend cap sums over,
and one row per node it ran through."""

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.chat.enums import ChatOutcome
from app.core.db.schema import BaseSchema


class ChatRequest(BaseSchema):
    """One handled question: how it ended, how long it lived, what it cost, and what failed.
    A spend cap sums tokens over a recent window; the index keeps that off a full scan.
    The path it took, node by node, is in chat_request_nodes."""

    __tablename__ = "chat_requests"
    __table_args__ = (Index("ix_chat_requests_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[str | None]
    question: Mapped[str]
    outcome: Mapped[ChatOutcome]
    model: Mapped[str]
    total_ms: Mapped[int]
    sources: Mapped[int]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    error: Mapped[str | None]

    nodes: Mapped[list["ChatRequestNode"]] = relationship(
        cascade="all, delete-orphan", order_by="ChatRequestNode.position"
    )


class ChatRequestNode(BaseSchema):
    """One step of a request's path: which node, in what order, how long it took, and the
    tokens it used if it called a model. The node is a plain string: names will grow with
    tools, and a String column takes new ones without a migration."""

    __tablename__ = "chat_request_nodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_request_id: Mapped[int] = mapped_column(
        ForeignKey("chat_requests.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int]
    node: Mapped[str]
    ms: Mapped[int]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
