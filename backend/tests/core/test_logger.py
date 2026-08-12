"""Tests for structured logging."""

import json
import logging
from typing import Any, cast

from app.core import logger as logger_module
from app.core.config import Environment
from app.core.logger import (
    _AppFormatter,
    _build_formatter,
    _ContextFilter,
    request_id_var,
    setup_logging,
)


def _record(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_context_filter_binds_request_id() -> None:
    token = request_id_var.set("abc123def456")
    try:
        record = _record()
        assert _ContextFilter().filter(record) is True
        assert cast(Any, record).request_id == "abc123def456"
    finally:
        request_id_var.reset(token)


def test_context_filter_binds_none_outside_request() -> None:
    record = _record()
    _ContextFilter().filter(record)
    assert cast(Any, record).request_id is None


def test_dev_formatter_prefixes_short_request_id() -> None:
    record = _record()
    cast(Any, record).request_id = "abc123def456"
    assert _AppFormatter().format(record).endswith("INFO    [abc123de] hello")


def test_dev_formatter_without_request_id() -> None:
    record = _record()
    cast(Any, record).request_id = None
    assert _AppFormatter().format(record).endswith("INFO    hello")


def test_prod_formatter_emits_json(monkeypatch) -> None:
    monkeypatch.setattr(logger_module.config, "ENVIRONMENT", Environment.PROD)
    formatter = _build_formatter()
    record = _record()
    cast(Any, record).request_id = "abc123"
    body = json.loads(formatter.format(record))
    assert body["message"] == "hello"
    assert body["request_id"] == "abc123"
    assert body["level"] == "INFO"
    assert body["logger"] == "app.test"


def test_dev_formatter_selected_outside_prod() -> None:
    assert isinstance(_build_formatter(), _AppFormatter)


def test_setup_logging_is_idempotent() -> None:
    """Every entrypoint calls it, and a second call must not double every log line."""
    setup_logging()
    setup_logging()
    assert len(logging.getLogger("app").handlers) == 1
    assert len(logging.getLogger("uvicorn").handlers) == 1
