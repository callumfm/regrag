"""Clock accessors, kept in one place so tests can pin the current time."""

from datetime import UTC, date, datetime


def utc_now() -> datetime:
    """The current UTC instant."""
    return datetime.now(UTC)


def utc_today() -> date:
    """The current UTC date."""
    return utc_now().date()
