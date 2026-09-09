"""Chat tools: definitions the model sees, argument validation, and dispatch."""

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.chat.enums import ChatStepStatus, ToolStep
from app.chat.models import ToolCall
from app.chat.tools import (
    build_call_step,
    describe_call,
    is_insufficient_context,
    run_tool_call,
    tool_definitions,
)
from app.core.config import config
from app.core.llm import LLMError
from app.retrieval.models import ReferenceTarget, SearchFilters, SearchRequest
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


def test_definitions_name_every_tool_with_parameter_schemas(monkeypatch):
    monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", True)
    names = [d["function"]["name"] for d in tool_definitions()]
    assert names == ["search", "follow_reference", "insufficient_context"]
    for definition in tool_definitions():
        assert definition["type"] == "function"
        assert "properties" in definition["function"]["parameters"]
        assert definition["function"]["description"]


def test_insufficient_context_is_left_off_the_surface_when_the_switch_is_off(monkeypatch):
    monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", False)
    names = [d["function"]["name"] for d in tool_definitions()]
    assert names == ["search", "follow_reference"]


def test_insufficient_context_asks_for_a_reason(monkeypatch):
    monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", True)
    tool = next(d for d in tool_definitions() if d["function"]["name"] == "insufficient_context")
    assert tool["function"]["parameters"]["required"] == ["reason"]


def test_only_a_call_to_insufficient_context_is_one():
    assert is_insufficient_context(ToolCall(name="insufficient_context", args={"reason": "none"}))
    assert not is_insufficient_context(ToolCall(name="search", args={"query": "insufficient"}))


async def test_an_insufficient_context_call_fetches_nothing():
    """The call is its own result: the tools node reads the reason off it, so running it
    adds no chunk to the context."""
    call = ToolCall(name="insufficient_context", args={"reason": "nothing bears on it"})

    assert await run_tool_call(call) == ()


def test_an_insufficient_context_call_records_its_reason_as_the_steps_subject():
    call = ToolCall(name="insufficient_context", args={"reason": "nothing bears on it"})

    step = build_call_step(call)

    assert step.step is ToolStep.INSUFFICIENT_CONTEXT
    assert step.subject == "nothing bears on it"


async def test_search_call_dispatches_with_filters_and_the_assess_limit(monkeypatch):
    requests: list[SearchRequest] = []

    async def fake_search(session, request):
        requests.append(request)
        return (search_result(),)

    monkeypatch.setattr("app.chat.tools.search", fake_search)
    call = ToolCall(name="search", args={"query": "penalties", "celex": "32023R1805"})

    found = await run_tool_call(call)

    assert found == (search_result(),)
    assert requests == [
        SearchRequest(
            query="penalties",
            filters=SearchFilters(celex="32023R1805"),
            limit=config.ASSESS_SEARCH_LIMIT,
        )
    ]


async def test_follow_reference_call_dispatches_to_the_named_division(monkeypatch):
    targets: list[ReferenceTarget] = []

    async def fake_follow(session, target):
        targets.append(target)
        return (search_result(id=7),)

    monkeypatch.setattr("app.chat.tools.follow_reference", fake_follow)
    call = ToolCall(name="follow_reference", args={"celex": "32023R1805", "article": "6"})

    found = await run_tool_call(call)

    assert found == (search_result(id=7),)
    assert targets == [ReferenceTarget(celex="32023R1805", article="6")]


async def test_an_unknown_tool_name_returns_nothing():
    assert await run_tool_call(ToolCall(name="check_in_force", args={})) == ()


async def test_invalid_arguments_return_nothing():
    call = ToolCall(name="search", args={"limit": 5})
    assert await run_tool_call(call) == ()


async def test_an_act_without_a_division_returns_nothing():
    call = ToolCall(name="follow_reference", args={"celex": "32023R1805"})
    assert await run_tool_call(call) == ()


async def test_a_search_call_that_raises_llmerror_returns_nothing(monkeypatch):
    async def failing_search(session, request):
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.chat.tools.search", failing_search)
    call = ToolCall(name="search", args={"query": "penalties"})

    assert await run_tool_call(call) == ()


async def test_a_search_call_that_raises_a_database_error_returns_nothing(monkeypatch):
    async def failing_search(session, request):
        raise SQLAlchemyError("connection lost")

    monkeypatch.setattr("app.chat.tools.search", failing_search)
    call = ToolCall(name="search", args={"query": "penalties"})

    assert await run_tool_call(call) == ()


async def test_hits_below_the_retrieval_bar_are_not_added_to_the_context(monkeypatch):
    """The gate refuses to answer from junk; the loop may not smuggle the same junk in."""

    async def junk_search(session, request):
        return (search_result(cosine_similarity=0.2, reranker_relevance=0.3),)

    monkeypatch.setattr("app.chat.tools.search", junk_search)
    call = ToolCall(name="search", args={"query": "best pizza topping"})

    assert await run_tool_call(call) == ()


async def test_a_long_division_is_capped_to_the_follow_limit(monkeypatch):
    """One wide annex would otherwise spend the round's whole budget on a single call."""
    monkeypatch.setattr(config, "ASSESS_FOLLOW_LIMIT", 2)

    async def wide_follow(session, target):
        return tuple(search_result(id=n) for n in range(1, 6))

    monkeypatch.setattr("app.chat.tools.follow_reference", wide_follow)
    call = ToolCall(name="follow_reference", args={"celex": "32023R1805", "annex": "I"})

    found = await run_tool_call(call)

    assert tuple(chunk.id for chunk in found) == (1, 2)


async def test_a_call_that_fails_on_the_database_leaves_the_next_call_working(monkeypatch):
    """Each call owns its session, so the rollback the failing one owes is not left for the
    call after it to trip over."""
    attempts: list[str] = []

    async def flaky_search(session, request):
        attempts.append(request.query)
        if len(attempts) == 1:
            raise SQLAlchemyError("connection lost")
        return (search_result(id=9),)

    monkeypatch.setattr("app.chat.tools.search", flaky_search)

    first = await run_tool_call(ToolCall(name="search", args={"query": "first"}))
    second = await run_tool_call(ToolCall(name="search", args={"query": "second"}))

    assert first == ()
    assert second == (search_result(id=9),)


class TestDescribeCall:
    def test_search_is_described_by_its_query(self):
        call = ToolCall(name="search", args={"query": "verification of monitoring plans"})
        assert describe_call(call) == "verification of monitoring plans"

    def test_every_argument_given_is_described_in_order(self):
        call = ToolCall(name="follow_reference", args={"celex": "32023R1805", "article": "2"})
        assert describe_call(call) == "32023R1805 · 2"

    def test_arguments_left_out_are_not_described(self):
        call = ToolCall(name="search", args={"query": "scope", "celex": None})
        assert describe_call(call) == "scope"

    def test_a_call_with_no_arguments_has_nothing_to_say(self):
        assert describe_call(ToolCall(name="search")) is None


class TestBuildCallStep:
    def test_the_starting_and_settled_frames_name_the_same_step_and_subject(self):
        call = ToolCall(name="search", args={"query": "scope"})
        running = build_call_step(call, status=ChatStepStatus.RUNNING)
        settled = build_call_step(call, ms=40)

        assert (running.step, running.subject) == (settled.step, settled.subject)
        assert (running.status, running.ms) == (ChatStepStatus.RUNNING, 0)
        assert (settled.status, settled.ms) == (ChatStepStatus.COMPLETED, 40)

    def test_a_tool_the_surface_lacks_is_recorded_as_unknown(self):
        assert build_call_step(ToolCall(name="teleport")).step is ToolStep.UNKNOWN
