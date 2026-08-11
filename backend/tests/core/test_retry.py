"""The shared transient-retry policy: what it retries, how often, and what it re-raises."""

import pytest

from app.core.retry import MAX_ATTEMPTS, transient_retry


class Transient(Exception):
    """Stand-in for a failure the predicate accepts."""


class Permanent(Exception):
    """Stand-in for a failure the predicate rejects."""


@pytest.fixture
def flaky(defuse_retry):
    """Build a callable that raises the queued errors in turn, then returns 'ok'."""

    def _make(errors):
        calls: list[int] = []

        @transient_retry(lambda exc: isinstance(exc, Transient))
        def _call() -> str:
            calls.append(len(calls))
            if calls[-1] < len(errors):
                raise errors[calls[-1]]
            return "ok"

        return defuse_retry(_call), calls

    return _make


def test_max_attempts_is_three():
    assert MAX_ATTEMPTS == 3


def test_retries_until_the_call_succeeds(flaky):
    call, calls = flaky([Transient()])
    assert call() == "ok"
    assert len(calls) == 2


def test_gives_up_after_max_attempts_and_reraises_the_original(flaky):
    call, calls = flaky([Transient()] * MAX_ATTEMPTS)
    with pytest.raises(Transient):
        call()
    assert len(calls) == MAX_ATTEMPTS


def test_does_not_retry_what_the_predicate_rejects(flaky):
    call, calls = flaky([Permanent()])
    with pytest.raises(Permanent):
        call()
    assert len(calls) == 1


@pytest.mark.anyio
async def test_retries_a_coroutine_the_same_way_as_a_function(defuse_retry):
    """Pins tenacity's coroutine detection: without it a wrapped async call never retries."""
    calls: list[int] = []

    @transient_retry(lambda exc: isinstance(exc, Transient))
    async def _call() -> str:
        calls.append(len(calls))
        if calls[-1] < 1:
            raise Transient
        return "ok"

    assert await defuse_retry(_call)() == "ok"
    assert len(calls) == 2
