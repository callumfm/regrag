"""Shared outbound-HTTP policy: headers, timeout, transport-level retries."""

import httpx

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def http_client(timeout: float = 30.0) -> httpx.Client:
    """Client with default headers, timeout, connect retries, and redirect following."""
    return httpx.Client(
        headers=DEFAULT_HEADERS,
        timeout=timeout,
        transport=httpx.HTTPTransport(retries=3),
        follow_redirects=True,
    )
