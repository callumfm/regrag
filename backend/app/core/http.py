"""Shared outbound-HTTP policy: headers, timeout, transport- and response-level retries."""

import logging

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
MAX_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def http_client(timeout: float = 30.0) -> httpx.Client:
    """Client with default headers, timeout, connect retries, and redirect following."""
    return httpx.Client(
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        transport=httpx.HTTPTransport(retries=3),
        follow_redirects=True,
    )


def _is_transient(exc: BaseException) -> bool:
    """Network errors and retryable status codes; client errors are permanent."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


def _log_retry(state: RetryCallState) -> None:
    logger.warning(
        "Retrying %s (attempt %d/%d) after transient error: %s",
        getattr(state.fn, "__name__", "request"),
        state.attempt_number,
        MAX_ATTEMPTS,
        state.outcome.exception() if state.outcome else None,
    )


transient_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=_log_retry,
    reraise=True,
)
"""Decorator retrying transient HTTP failures with exponential backoff."""
