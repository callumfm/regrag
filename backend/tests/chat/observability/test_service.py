"""Chat run recording: the row, the log line, and a write that fails quietly."""

from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ChatOutcome
from app.chat.observability import service
from app.chat.observability.models import StreamStats
from app.chat.observability.schemas import ChatRun
from app.chat.observability.service import chat_run, log_run, record_run
from app.core.config import config
from app.core.logger import request_id_var
from tests.chat.conftest import USAGE

pytestmark = pytest.mark.anyio


def finished_stats() -> StreamStats:
    return StreamStats(retrieve_ms=120, ttft_ms=800, sources=6, usage=USAGE)


@pytest.fixture
def stats_lines(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    """Capture (rendered message, extra) for each stats line the service logs."""
    lines: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        service.logger, "info", lambda msg, *args, extra: lines.append((msg % args, extra))
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


def test_chat_run_row_reads_the_stats_and_the_request_context():
    token = request_id_var.set("abc123")
    try:
        run = chat_run(finished_stats(), ChatOutcome.DONE)
    finally:
        request_id_var.reset(token)

    assert run.request_id == "abc123"
    assert run.outcome is ChatOutcome.DONE
    assert run.model == config.CHAT_MODEL
    assert (run.retrieve_ms, run.ttft_ms, run.sources) == (120, 800, 6)
    assert (run.input_tokens, run.output_tokens) == (1500, 40)
    assert run.total_ms >= 0


def test_chat_run_row_without_usage_or_timings_is_null_there():
    run = chat_run(StreamStats(), ChatOutcome.ERROR)
    assert (run.retrieve_ms, run.ttft_ms, run.input_tokens, run.output_tokens) == (None,) * 4
    assert run.sources == 0


def test_log_line_carries_every_field(stats_lines):
    log_run(chat_run(finished_stats(), ChatOutcome.DONE))

    [(message, extra)] = stats_lines
    assert message.startswith("chat done - retrieve 120ms, first token 800ms, total ")
    assert message.endswith("ms, 6 sources, 1500/40 tokens")
    assert extra["outcome"] is ChatOutcome.DONE
    assert extra["input_tokens"] == 1500


async def test_record_run_persists_one_row(own_session: AsyncSession, stats_lines):
    await record_run(finished_stats(), ChatOutcome.DONE)

    [run] = (await own_session.scalars(select(ChatRun))).all()
    assert run.outcome is ChatOutcome.DONE
    assert run.output_tokens == 40
    assert run.created_at is not None
    assert len(stats_lines) == 1


async def test_failed_write_is_logged_not_raised(monkeypatch, stats_lines):
    @asynccontextmanager
    async def broken_session(**kwargs):
        raise ConnectionError("database away")
        yield

    errors: list[str] = []
    monkeypatch.setattr(service, "get_session", broken_session)
    monkeypatch.setattr(service.logger, "exception", lambda msg, *args: errors.append(msg))

    await record_run(finished_stats(), ChatOutcome.DONE)

    assert errors == ["chat run not recorded"]
    assert len(stats_lines) == 1
