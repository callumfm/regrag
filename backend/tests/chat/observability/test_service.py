"""Chat request recording: the row, the log line, and a write that fails quietly."""

from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import select
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


@pytest.fixture
def stats_lines(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    """Capture (rendered message, extra) for each stats line the service logs."""
    lines: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        service.logger, "info", lambda msg, fields, extra: lines.append((msg % fields, extra))
    )
    return lines


@pytest.fixture
def own_session(monkeypatch, db_session: AsyncSession) -> AsyncSession:
    """Point the service's session at the test transaction, so its row rolls back with it."""

    @asynccontextmanager
    async def test_session(**kwargs):
        yield db_session

    monkeypatch.setattr(service, "get_session", test_session)
    return db_session


async def test_recorded_row_reads_the_stats_and_the_request_context(
    own_session: AsyncSession, stats_lines
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
    assert len(stats_lines) == 1


async def test_recorded_row_without_usage_or_timings_is_null_there(
    own_session: AsyncSession, stats_lines
):
    await record_request("q", RequestStats(), ChatOutcome.ERROR)

    [row] = (await own_session.scalars(select(ChatRequest))).all()
    assert (row.retrieve_ms, row.ttft_ms, row.input_tokens, row.output_tokens) == (None,) * 4
    assert row.sources == 0


async def test_log_line_carries_every_field(own_session: AsyncSession, stats_lines):
    await record_request("q", finished_stats(), ChatOutcome.DONE)

    [(message, extra)] = stats_lines
    assert message.startswith("chat done - retrieve 120ms, first token 800ms, total ")
    assert message.endswith("ms, 6 sources, 1500/40 tokens")
    assert extra["outcome"] is ChatOutcome.DONE
    assert extra["input_tokens"] == 1500


async def test_failed_write_is_logged_not_raised(monkeypatch, stats_lines):
    @asynccontextmanager
    async def broken_session(**kwargs):
        raise ConnectionError("database away")
        yield

    errors: list[str] = []
    monkeypatch.setattr(service, "get_session", broken_session)
    monkeypatch.setattr(service.logger, "exception", lambda msg, *args: errors.append(msg))

    await record_request("q", finished_stats(), ChatOutcome.DONE)

    assert errors == ["chat request not recorded"]
    assert len(stats_lines) == 1
