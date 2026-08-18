"""Chat request recording: the row, the log line, and a write that fails quietly."""

import logging
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ChatOutcome
from app.chat.observability import service
from app.chat.observability.models import RequestStats
from app.chat.observability.schemas import ChatRequest
from app.chat.observability.service import record_request
from app.core.config import config
from app.core.logger import request_id_var
from tests.chat.conftest import USAGE

pytestmark = pytest.mark.anyio


def finished_stats() -> RequestStats:
    return RequestStats(retrieve_ms=120, ttft_ms=800, sources=6, usage=USAGE)


def stats_lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The stats lines the service logged, with their extra fields as record attributes."""
    return [
        record
        for record in caplog.records
        if record.name == service.logger.name and record.levelno == logging.INFO
    ]


@pytest.fixture
def own_session(monkeypatch, db_session: AsyncSession) -> AsyncSession:
    """Point the service's session at the test transaction, so its row rolls back with it."""

    @asynccontextmanager
    async def test_session(**kwargs):
        yield db_session

    monkeypatch.setattr(service, "get_session", test_session)
    return db_session


async def test_recorded_row_reads_the_stats_and_the_request_context(
    own_session: AsyncSession, caplog
):
    token = request_id_var.set("abc123")
    try:
        await record_request("What must ships report?", finished_stats(), ChatOutcome.DONE)
    finally:
        request_id_var.reset(token)

    [row] = (await own_session.scalars(select(ChatRequest))).all()
    assert row.question == "What must ships report?"
    assert row.request_id == "abc123"
    assert row.outcome is ChatOutcome.DONE
    assert row.model == config.CHAT_MODEL
    assert (row.retrieve_ms, row.ttft_ms, row.sources) == (120, 800, 6)
    assert (row.input_tokens, row.output_tokens) == (1500, 40)
    assert row.total_ms >= 0
    assert row.created_at is not None
    assert len(stats_lines(caplog)) == 1


async def test_recorded_row_without_usage_or_timings_is_null_there(
    own_session: AsyncSession, caplog
):
    await record_request("q", RequestStats(), ChatOutcome.ERROR)

    [row] = (await own_session.scalars(select(ChatRequest))).all()
    assert (row.retrieve_ms, row.ttft_ms, row.input_tokens, row.output_tokens) == (None,) * 4
    assert row.sources == 0


async def test_log_line_carries_every_field(own_session: AsyncSession, caplog):
    await record_request("q", finished_stats(), ChatOutcome.DONE)

    [record] = stats_lines(caplog)
    message = record.getMessage()
    assert message.startswith("chat done - retrieve 120ms, first token 800ms, total ")
    assert message.endswith("ms, 6 sources, 1500/40 tokens")
    assert record.__dict__["outcome"] is ChatOutcome.DONE
    assert record.__dict__["input_tokens"] == 1500


async def test_failed_write_is_logged_not_raised(monkeypatch, caplog):
    @asynccontextmanager
    async def broken_session(**kwargs):
        raise OperationalError("connect", {}, ConnectionRefusedError("database away"))
        yield

    monkeypatch.setattr(service, "get_session", broken_session)

    await record_request("q", finished_stats(), ChatOutcome.DONE)

    assert len(stats_lines(caplog)) == 1
    [error] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error.getMessage() == "chat request not recorded"
    assert error.exc_info is not None


async def test_a_refused_request_is_recorded_as_such(own_session: AsyncSession, caplog):
    """The gate's outcome fits the column as migrated: refused is no longer than aborted."""
    await record_request("best pizza topping?", RequestStats(retrieve_ms=90), ChatOutcome.REFUSED)

    [row] = (await own_session.scalars(select(ChatRequest))).all()
    assert row.outcome is ChatOutcome.REFUSED
    assert (row.retrieve_ms, row.ttft_ms, row.sources, row.input_tokens) == (90, None, 0, None)
    assert stats_lines(caplog)[0].getMessage().startswith("chat refused - retrieve 90ms")
