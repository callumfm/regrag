"""The tool surface: what the model is shown, how a call is dispatched, and the step
one records."""

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.chat.enums import ChatStepStatus, ToolStep
from app.chat.toolbox.models import ToolCall
from app.chat.toolbox.service import build_call_step, describe_call, run_tool_call, tool_definitions
from app.core.config import config
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


def test_definitions_name_every_tool_with_parameter_schemas(monkeypatch):
    monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", True)
    names = [d["function"]["name"] for d in tool_definitions()]
    assert names == ["search", "follow_reference", "refuse"]
    for definition in tool_definitions():
        assert definition["type"] == "function"
        assert "properties" in definition["function"]["parameters"]
        assert definition["function"]["description"]


def test_refuse_is_left_off_the_surface_when_the_switch_is_off(monkeypatch):
    monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", False)
    names = [d["function"]["name"] for d in tool_definitions()]
    assert names == ["search", "follow_reference"]


async def test_an_unknown_tool_name_returns_nothing():
    assert await run_tool_call(ToolCall(name="check_in_force", args={})) == ()


async def test_invalid_arguments_return_nothing():
    call = ToolCall(name="search", args={"limit": 5})
    assert await run_tool_call(call) == ()


async def test_a_call_that_fails_on_the_database_leaves_the_next_call_working(monkeypatch):
    """Each call owns its session, so the rollback the failing one owes is not left for the
    call after it to trip over."""
    attempts: list[str] = []

    async def flaky_search(session, request):
        attempts.append(request.query)
        if len(attempts) == 1:
            raise SQLAlchemyError("connection lost")
        return (search_result(id=9),)

    monkeypatch.setattr("app.chat.toolbox.tools.search.search", flaky_search)

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
