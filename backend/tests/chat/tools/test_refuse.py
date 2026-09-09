"""refuse: the explanation it asks for, that it fetches nothing, and the step it leaves."""

import pytest

from app.chat.enums import ToolStep
from app.chat.models import ToolCall
from app.chat.toolbox import build_call_step, run_tool_call, tool_definitions
from app.chat.tools.refuse import is_refusal
from app.core.config import config

pytestmark = pytest.mark.anyio


def test_refuse_asks_for_an_explanation(monkeypatch):
    monkeypatch.setattr(config, "ASSESS_MAY_REFUSE", True)
    tool = next(d for d in tool_definitions() if d["function"]["name"] == "refuse")
    assert tool["function"]["parameters"]["required"] == ["explanation"]


def test_only_a_call_to_refuse_is_one():
    assert is_refusal(ToolCall(name="refuse", args={"explanation": "none"}))
    assert not is_refusal(ToolCall(name="search", args={"query": "refuse"}))


async def test_a_refuse_call_fetches_nothing():
    """The call is its own result: the tools node reads the explanation off it, so running
    it adds no chunk to the context."""
    call = ToolCall(name="refuse", args={"explanation": "nothing bears on it"})

    assert await run_tool_call(call) == ()


def test_a_refuse_call_records_its_explanation_as_the_steps_subject():
    call = ToolCall(name="refuse", args={"explanation": "nothing bears on it"})

    step = build_call_step(call)

    assert step.step is ToolStep.REFUSE
    assert step.subject == "nothing bears on it"
