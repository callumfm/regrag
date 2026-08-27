"""Chat tools: definitions the model sees, argument validation, and dispatch."""

import pytest

from app.chat.models import ToolCall
from app.chat.tools import run_tool_call, tool_definitions
from app.core.config import config
from app.retrieval.models import ReferenceTarget, SearchFilters, SearchRequest
from tests.conftest import search_result

pytestmark = pytest.mark.anyio


def test_definitions_name_both_tools_with_parameter_schemas():
    definitions = tool_definitions()

    names = [d["function"]["name"] for d in definitions]
    assert names == ["search", "follow_reference"]
    for definition in definitions:
        assert definition["type"] == "function"
        assert "properties" in definition["function"]["parameters"]
        assert definition["function"]["description"]


async def test_search_call_dispatches_with_filters_and_the_gather_limit(monkeypatch):
    requests: list[SearchRequest] = []

    async def fake_search(session, request):
        requests.append(request)
        return (search_result(),)

    monkeypatch.setattr("app.chat.tools.search", fake_search)
    call = ToolCall(name="search", args={"query": "penalties", "celex": "32023R1805"})

    found = await run_tool_call(None, call)

    assert found == (search_result(),)
    assert requests == [
        SearchRequest(
            query="penalties",
            filters=SearchFilters(celex="32023R1805"),
            limit=config.GATHER_SEARCH_LIMIT,
        )
    ]


async def test_follow_reference_call_dispatches_to_the_named_division(monkeypatch):
    targets: list[ReferenceTarget] = []

    async def fake_follow(session, target):
        targets.append(target)
        return (search_result(id=7),)

    monkeypatch.setattr("app.chat.tools.follow_reference", fake_follow)
    call = ToolCall(name="follow_reference", args={"celex": "32023R1805", "article": "6"})

    found = await run_tool_call(None, call)

    assert found == (search_result(id=7),)
    assert targets == [ReferenceTarget(celex="32023R1805", article="6")]


async def test_an_unknown_tool_name_returns_nothing():
    assert await run_tool_call(None, ToolCall(name="check_in_force", args={})) == ()


async def test_invalid_arguments_return_nothing():
    call = ToolCall(name="search", args={"limit": 5})
    assert await run_tool_call(None, call) == ()


async def test_an_act_without_a_division_returns_nothing():
    call = ToolCall(name="follow_reference", args={"celex": "32023R1805"})
    assert await run_tool_call(None, call) == ()
