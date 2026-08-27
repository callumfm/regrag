"""The runner: the grid, one case through retrieve alone, and the config-override run loop."""

import logging

import pytest

from app.chat.enums import ChatNode, ChatOutcome
from app.core.config import config
from app.core.llm import LLMError
from app.evals.tune import service
from app.evals.tune.models import GridPoint
from app.evals.tune.service import build_grid, run_grid, run_retrieval
from tests.conftest import search_result
from tests.evals.conftest import eval_case, eval_dataset, eval_result

pytestmark = pytest.mark.anyio


def test_build_grid_varies_each_param_alone() -> None:
    points = build_grid({"CHAT_SOURCES": (3, 8), "RERANK_ENABLED": (False,)})

    assert points == (
        GridPoint(overrides={"CHAT_SOURCES": 3}),
        GridPoint(overrides={"CHAT_SOURCES": 8}),
        GridPoint(overrides={"RERANK_ENABLED": False}),
    )


def test_build_grid_drops_a_value_equal_to_baseline() -> None:
    baseline = config.CHAT_SOURCES

    points = build_grid({"CHAT_SOURCES": (baseline, baseline + 1)})

    assert points == (GridPoint(overrides={"CHAT_SOURCES": baseline + 1}),)


@pytest.fixture
def found_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search finds the authored reference; expansion is a database walk covered in
    tests/retrieval, so it is switched off."""

    async def fake_search(session, request):
        return (search_result(),)

    monkeypatch.setattr(config, "EXPAND_SECTIONS", False)
    monkeypatch.setattr("app.chat.graph.search", fake_search)


async def test_run_retrieval_drives_the_retrieve_node_alone(found_context: None) -> None:
    result = await run_retrieval(eval_case())

    assert result.state.hits == (search_result(),)
    assert result.state.sources == (search_result(),)
    assert [n.node for n in result.state.nodes] == [ChatNode.RETRIEVE]
    assert result.state.token_totals() == (None, None)
    assert result.state.total_ms is not None


async def test_a_case_retrieve_raises_on_is_recorded_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def failing_search(session, request):
        raise LLMError("embedding call failed")

    monkeypatch.setattr("app.chat.graph.search", failing_search)

    with caplog.at_level(logging.WARNING):
        result = await run_retrieval(eval_case())

    assert result.state.error == "embedding call failed"
    assert result.state.outcome is ChatOutcome.ERROR


async def test_run_grid_applies_each_point_and_restores_baseline_between(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, bool]] = []

    async def fake_run(case):
        seen.append((config.CHAT_SOURCES, config.RERANK_ENABLED))
        return eval_result(case=case)

    monkeypatch.setattr(service, "run_retrieval", fake_run)
    baseline_sources = config.CHAT_SOURCES
    baseline_rerank = config.RERANK_ENABLED
    dataset = eval_dataset(eval_case(id="one"), eval_case(id="two"))
    points = (
        GridPoint(overrides={"CHAT_SOURCES": baseline_sources + 1}),
        GridPoint(overrides={"RERANK_ENABLED": not baseline_rerank}),
    )

    run = await run_grid(dataset, points)

    assert seen == [
        (baseline_sources, baseline_rerank),
        (baseline_sources, baseline_rerank),
        (baseline_sources + 1, baseline_rerank),
        (baseline_sources + 1, baseline_rerank),
        (baseline_sources, not baseline_rerank),
        (baseline_sources, not baseline_rerank),
    ]
    assert config.CHAT_SOURCES == baseline_sources
    assert config.RERANK_ENABLED == baseline_rerank
    assert run.baseline.point == GridPoint()
    assert [tp.point for tp in run.results[1:]] == list(points)
    assert run.dataset_sha == dataset.sha256


async def test_run_grid_filters_cases_by_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    async def fake_run(case):
        ran.append(case.id)
        return eval_result(case=case)

    monkeypatch.setattr(service, "run_retrieval", fake_run)
    dataset = eval_dataset(eval_case(id="fueleu-one"), eval_case(id="mrv-one"))

    run = await run_grid(dataset, (), pattern="fueleu")

    assert ran == ["fueleu-one"]
    assert run.case_pattern == "fueleu"
