"""Shared outbound-HTTP policy: headers, timeout, pacing, and transient-failure retries."""

import time
from collections.abc import Callable

import httpx

from app.core.retry import transient_retry

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def pace_requests(seconds: float) -> Callable[[httpx.Request], None]:
    """Hook spacing every request this client makes, keeping last-call time scoped to it."""
    last: float | None = None

    def pace(request: httpx.Request) -> None:
        nonlocal last
        if last is not None and (remaining := seconds - (time.monotonic() - last)) > 0:
            time.sleep(remaining)
        last = time.monotonic()

    return pace


def http_client(timeout: float = 30.0, pace_seconds: float | None = None) -> httpx.Client:
    """Client with default headers, timeout, redirect following, and optional request pacing."""
    return httpx.Client(
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"request": [pace_requests(pace_seconds)]} if pace_seconds else {},
    )


def _is_transient(exc: BaseException) -> bool:
    """Network errors and retryable status codes; client errors are permanent."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


http_retry = transient_retry(_is_transient)
"""Decorator retrying transient HTTP failures with exponential backoff."""
