"""refuse: the explanation it asks for, that it fetches nothing, and the step it leaves."""

import pytest

from app.chat.enums import RefusalReason, ToolStep
from app.chat.toolbox.models import ToolCall
from app.chat.toolbox.service import build_call_step, run_tool_call
from app.chat.toolbox.tools.refuse import REFUSE, is_refusal, refusal_from

pytestmark = pytest.mark.anyio


def test_refuse_asks_for_an_explanation():
    assert REFUSE.definition()["function"]["parameters"]["required"] == ["explanation"]


def test_a_refuse_call_carries_its_explanation_as_the_refusal():
    refusal = refusal_from(ToolCall(name="refuse", args={"explanation": "nothing bears on it"}))

    assert refusal is not None
    assert (refusal.reason, refusal.explanation) == (
        RefusalReason.INSUFFICIENT_CONTEXT,
        "nothing bears on it",
    )


def test_a_refuse_call_without_an_explanation_is_recorded_as_empty():
    refusal = refusal_from(ToolCall(name="refuse", args={}))

    assert refusal is not None
    assert refusal.explanation == ""


def test_a_fetch_carries_no_refusal():
    assert refusal_from(ToolCall(name="search", args={"query": "refuse"})) is None


def test_only_a_call_to_refuse_is_one():
    assert is_refusal(ToolCall(name="refuse", args={"explanation": "none"}))
    assert not is_refusal(ToolCall(name="search", args={"query": "refuse"}))


async def test_a_refuse_call_fetches_nothing():
    """The call is its own result: assess_tools reads the explanation off it, so running
    it adds no chunk to the context."""
    call = ToolCall(name="refuse", args={"explanation": "nothing bears on it"})

    assert await run_tool_call(call) == ()


def test_a_refuse_call_records_its_explanation_as_the_steps_subject():
    call = ToolCall(name="refuse", args={"explanation": "nothing bears on it"})

    step = build_call_step(call)

    assert step.step is ToolStep.REFUSE
    assert step.subject == "nothing bears on it"
