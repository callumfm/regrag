"""Chat request recording: the row, its node rows, and the log line."""

import logging

import pytest
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import service
from app.chat.enums import ChatNode, ChatOutcome
from app.chat.models import ChatNodeResult, ChatState
from app.chat.schemas import ChatRequest, ChatRequestNode
from app.chat.service import create_chat_request
from app.core.config import config
from app.core.logger import request_id_var
from tests.chat.conftest import USAGE
from tests.conftest import retrieved_chunk

pytestmark = pytest.mark.anyio


def answered_state() -> ChatState:
    """A state as the graph leaves it once an answer has been synthesized."""
    return ChatState(
        question="What must ships report?",
        nodes=(
            ChatNodeResult(node=ChatNode.RETRIEVE, ms=120),
            ChatNodeResult.from_usage(ChatNode.SYNTHESIZE, 1300, USAGE),
        ),
        sources=tuple(retrieved_chunk(id=n) for n in range(6)),
        answer="Ships must report [1].",
        total_ms=1500,
    )


def node_rows() -> Select[tuple[ChatRequestNode]]:
    """The node rows in path order — the order the relationship reads them in."""
    return select(ChatRequestNode).order_by(ChatRequestNode.position)


def stats_lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The stats lines the service logged, with their extra fields as record attributes."""
    return [
        record
        for record in caplog.records
        if record.name == service.logger.name and record.levelno == logging.INFO
    ]


async def test_recorded_row_reads_the_stats_and_the_request_context(
    db_session: AsyncSession, caplog
):
    token = request_id_var.set("abc123")
    try:
        await create_chat_request(db_session, answered_state())
    finally:
        request_id_var.reset(token)

    [row] = (await db_session.scalars(select(ChatRequest))).all()
    assert row.question == "What must ships report?"
    assert row.request_id == "abc123"
    assert row.outcome is ChatOutcome.DONE
    assert row.model == config.CHAT_MODEL
    assert row.sources == 6
    assert (row.input_tokens, row.output_tokens) == (1500, 40)
    assert row.total_ms == 1500
    assert row.error is None
    assert row.created_at is not None
    assert len(stats_lines(caplog)) == 1

    nodes = (await db_session.scalars(node_rows())).all()
    assert [(n.chat_request_id, n.position, n.node, n.ms) for n in nodes] == [
        (row.id, 0, "retrieve", 120),
        (row.id, 1, "synthesize", 1300),
    ]
    assert [(n.input_tokens, n.output_tokens) for n in nodes] == [(None, None), (1500, 40)]


async def test_failed_run_records_its_error_and_nulls_where_it_never_got(
    db_session: AsyncSession, caplog
):
    failed = ChatState(question="q", total_ms=40, error="embedding call failed")
    await create_chat_request(db_session, failed)

    [row] = (await db_session.scalars(select(ChatRequest))).all()
    assert row.outcome is ChatOutcome.ERROR
    assert row.error == "embedding call failed"
    assert (row.input_tokens, row.output_tokens) == (None, None)
    assert row.sources == 0
    assert (await db_session.scalars(select(ChatRequestNode))).all() == []


async def test_log_line_carries_the_stats_but_not_the_content(db_session: AsyncSession, caplog):
    await create_chat_request(db_session, answered_state())

    [record] = stats_lines(caplog)
    assert record.getMessage() == "chat done in 1500ms"
    assert record.__dict__["outcome"] == "done"
    assert record.__dict__["sources"] == 6
    assert record.__dict__["nodes"] == [
        {"node": "retrieve", "ms": 120, "input_tokens": None, "output_tokens": None},
        {"node": "synthesize", "ms": 1300, "input_tokens": 1500, "output_tokens": 40},
    ]
    assert "question" not in record.__dict__
    assert "answer" not in record.__dict__


async def test_a_refused_request_is_recorded_as_such(db_session: AsyncSession, caplog):
    """The gate's outcome fits the column as migrated: refused is no longer than aborted."""
    refused = ChatState(
        question="best pizza topping?",
        nodes=(
            ChatNodeResult(node=ChatNode.RETRIEVE, ms=90),
            ChatNodeResult(node=ChatNode.REFUSE, ms=0),
        ),
        total_ms=95,
    )
    await create_chat_request(db_session, refused)

    [row] = (await db_session.scalars(select(ChatRequest))).all()
    assert row.outcome is ChatOutcome.REFUSED
    assert (row.sources, row.input_tokens) == (0, None)
    nodes = (await db_session.scalars(node_rows())).all()
    assert [(n.node, n.ms) for n in nodes] == [("retrieve", 90), ("refuse", 0)]
    assert stats_lines(caplog)[0].getMessage() == "chat refused in 95ms"
