"""Shared HTTP client policy and the transient-retry decorator."""

import httpx
import pytest

from app.core.http import DEFAULT_HEADERS, download, http_client, http_retry, pace_requests


class FakeClock:
    """A monotonic clock that only moves when something sleeps, or a test advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Pacing's view of time, so tests assert on waits instead of serving them."""
    fake = FakeClock()
    monkeypatch.setattr("app.core.http.time.monotonic", fake.monotonic)
    monkeypatch.setattr("app.core.http.time.sleep", fake.sleep)
    return fake


def test_default_headers_carry_browser_user_agent():
    assert "Mozilla" in DEFAULT_HEADERS["User-Agent"]


def test_client_is_configured():
    with http_client(timeout=5.0) as client:
        assert client.headers["user-agent"] == DEFAULT_HEADERS["User-Agent"]
        assert client.follow_redirects is True
        assert client.timeout.read == 5.0


def test_client_does_not_pace_unless_asked():
    with http_client() as client:
        assert client.event_hooks["request"] == []


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

    @http_retry
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


def test_download_returns_the_response_body() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"hi"))
    )
    assert download(client, "https://example.test/doc") == b"hi"


def test_download_raises_on_a_client_error() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        download(client, "https://example.test/missing")


def paced_client(seconds: float) -> httpx.Client:
    """A client whose requests are paced and answered locally."""
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
        event_hooks={"request": [pace_requests(seconds)]},
    )


def test_pacing_lets_the_first_request_straight_through(clock: FakeClock) -> None:
    paced_client(1.0).get("https://example.test/doc")
    assert clock.slept == []


def test_pacing_waits_between_every_request_not_every_document(clock: FakeClock) -> None:
    """A document costs two requests — resolve, then download — and both hit the rate limit."""
    client = paced_client(1.0)
    for _ in range(3):
        client.get("https://example.test/doc")
    assert clock.slept == [1.0, 1.0]


def test_pacing_waits_only_for_what_is_left_of_the_interval(clock: FakeClock) -> None:
    client = paced_client(1.0)
    client.get("https://example.test/doc")
    clock.now += 0.4
    client.get("https://example.test/doc")
    assert clock.slept == [pytest.approx(0.6)]


def test_pacing_does_not_wait_when_the_interval_has_already_passed(clock: FakeClock) -> None:
    client = paced_client(1.0)
    client.get("https://example.test/doc")
    clock.now += 5.0
    client.get("https://example.test/doc")
    assert clock.slept == []


def test_each_client_paces_on_its_own_last_request(clock: FakeClock) -> None:
    """State lives with the client, so one client's requests never delay another's."""
    first, second = paced_client(1.0), paced_client(1.0)
    first.get("https://example.test/doc")
    second.get("https://example.test/doc")
    assert clock.slept == []
