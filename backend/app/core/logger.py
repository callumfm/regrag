"""Structured logging: JSON in prod, request-ID-prefixed text in dev."""

import logging
import sys

from pythonjsonlogger.json import JsonFormatter

from app.core.config import config
from app.core.context import request_id_var
from app.core.enums import Environment

logger = logging.getLogger("app")
logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger with shared handlers."""
    return logging.getLogger(name)


class _ContextFilter(logging.Filter):
    """Attach the request ID from the contextvar onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _AppFormatter(logging.Formatter):
    """Prepend request ID to dev-mode log lines when available."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None)
        if request_id:
            record.msg = f"[{request_id[:8]}] {record.msg}"
        return super().format(record)


def _build_formatter() -> logging.Formatter:
    if config.ENVIRONMENT == Environment.PROD:
        return JsonFormatter(
            "%(levelname)s %(name)s %(message)s %(request_id)s",
            rename_fields={"levelname": "level", "name": "logger"},
            timestamp=True,
        )
    return _AppFormatter(fmt="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(_build_formatter())
    _handler.addFilter(_ContextFilter())
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

uvicorn_logger = logging.getLogger("uvicorn")
for _existing in logger.handlers:
    if _existing not in uvicorn_logger.handlers:
        uvicorn_logger.addHandler(_existing)

logging.getLogger("uvicorn.access").disabled = True
