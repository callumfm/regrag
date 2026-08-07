"""Shared outbound-HTTP policy: headers, timeout, and transient-failure retries."""

import time

import httpx

from app.core.retry import transient_retry

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def http_client(timeout: float = 30.0) -> httpx.Client:
    """Client with default headers, timeout, and redirect following."""
    return httpx.Client(headers=DEFAULT_HEADERS, timeout=timeout, follow_redirects=True)


def _is_transient(exc: BaseException) -> bool:
    """Network errors and retryable status codes; client errors are permanent."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


http_retry = transient_retry(_is_transient)
"""Decorator retrying transient HTTP failures with exponential backoff."""


@http_retry
def download(client: httpx.Client, url: str) -> bytes:
    """Fetch a URL's bytes, retrying transient failures."""
    response = client.get(url)
    response.raise_for_status()
    return response.content


def pace(seconds: float) -> None:
    """Space out requests to an upstream host."""
    time.sleep(seconds)
