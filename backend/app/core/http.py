"""Shared outbound-HTTP policy: headers, timeout, pacing, and transient-failure retries."""

import asyncio
import time
from collections.abc import Callable, Coroutine, Mapping
from typing import Any

import httpx

from app.core.retry import transient_retry

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def pace_requests(
    delays: Mapping[str, float],
) -> Callable[[httpx.Request], Coroutine[Any, Any, None]]:
    """Hook spacing requests per host, so one host's crawl delay does not throttle another's."""
    last: dict[str, float] = {}

    async def pace(request: httpx.Request) -> None:
        host = request.url.host
        seconds = delays.get(host)
        if seconds is None:
            return
        previous = last.get(host)
        if previous is not None and (remaining := seconds - (time.monotonic() - previous)) > 0:
            await asyncio.sleep(remaining)
        last[host] = time.monotonic()

    return pace


def http_client(
    timeout: float = 30.0, delays: Mapping[str, float] | None = None
) -> httpx.AsyncClient:
    """Client with default headers, timeout, redirect following, and optional per-host pacing."""
    return httpx.AsyncClient(
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"request": [pace_requests(delays)]} if delays else {},
    )


def _is_transient(exc: BaseException) -> bool:
    """Network errors and retryable status codes; client errors are permanent."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


http_retry = transient_retry(_is_transient)
"""Decorator retrying transient HTTP failures with exponential backoff."""
