"""Shared HTTP client policy."""

from app.core.http import DEFAULT_HEADERS, http_client


def test_default_headers_carry_browser_user_agent():
    assert "Mozilla" in DEFAULT_HEADERS["User-Agent"]


def test_client_is_configured():
    with http_client(timeout=5.0) as client:
        assert client.headers["user-agent"] == DEFAULT_HEADERS["User-Agent"]
        assert client.follow_redirects is True
        assert client.timeout.read == 5.0
