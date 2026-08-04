"""Shared HTTP client policy and the transient-retry decorator."""

import httpx
import pytest

from app.core.http import DEFAULT_HEADERS, http_client, transient_retry


def test_default_headers_carry_browser_user_agent():
    assert "Mozilla" in DEFAULT_HEADERS["User-Agent"]


def test_client_is_configured():
    with http_client(timeout=5.0) as client:
        assert client.headers["user-agent"] == DEFAULT_HEADERS["User-Agent"]
        assert client.follow_redirects is True
        assert client.timeout.read == 5.0


def flaky_client(responses):
    """Client whose handler pops one queued response (or raises one queued error) per request."""
    calls = []

    def handler(request):
        calls.append(str(request.url))
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


@pytest.fixture
def get(defuse_retry):
    """The decorated unit under test, with tenacity's waits stripped."""

    @transient_retry
    def _get(client, url="https://example.eu/doc"):
        response = client.get(url)
        response.raise_for_status()
        return response

    return defuse_retry(_get)


def test_retries_retryable_status(get):
    client, calls = flaky_client([httpx.Response(503), httpx.Response(200, text="ok")])
    assert get(client).text == "ok"
    assert len(calls) == 2


def test_gives_up_after_max_attempts(get):
    client, calls = flaky_client([httpx.Response(503)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        get(client)
    assert len(calls) == 3


def test_does_not_retry_client_errors(get):
    client, calls = flaky_client([httpx.Response(404)])
    with pytest.raises(httpx.HTTPStatusError):
        get(client)
    assert len(calls) == 1


def test_retries_transport_errors(get):
    client, calls = flaky_client([httpx.ConnectError("refused"), httpx.Response(200, text="ok")])
    assert get(client).text == "ok"
    assert len(calls) == 2
