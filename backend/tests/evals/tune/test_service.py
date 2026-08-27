"""Grid tuning: parsing, point construction, and the config-override run loop."""

import logging

import pytest
from pydantic import ValidationError

from app.chat.enums import ChatNode, ChatOutcome
from app.core.config import config
from app.core.llm import LLMError
from app.evals.tune import service
from app.evals.tune.models import GridPoint
from app.evals.tune.service import (
    build_points,
    parse_settings,
    run_grid,
    run_retrieval,
    tunable_fields,
)
from tests.conftest import search_result
from tests.evals.conftest import eval_case, eval_dataset, eval_result

pytestmark = pytest.mark.anyio


def test_tunable_fields_cover_chat_and_retrieval_but_no_secret() -> None:
    fields = tunable_fields()

    assert "CHAT_SOURCES" in fields
    assert "RERANK_ENABLED" in fields
    assert "ANTHROPIC_API_KEY" not in fields
    assert "EMBED_MODEL" not in fields


def test_values_are_coerced_by_their_field_types() -> None:
    parsed = parse_settings(["CHAT_SOURCES=3,8", "RERANK_ENABLED=false"])

    assert parsed == {"CHAT_SOURCES": [3, 8], "RERANK_ENABLED": [False]}


def test_an_out_of_range_value_fails_before_any_run() -> None:
    with pytest.raises(ValidationError):
        parse_settings(["CHAT_SOURCES=0"])


def test_an_unknown_setting_is_named_with_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="CHAT_SOURCE is not a tunable setting"):
        parse_settings(["CHAT_SOURCE=3"])


def test_a_setting_given_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="CHAT_SOURCES"):
        parse_settings(["CHAT_SOURCES=3", "CHAT_SOURCES=8"])


def test_a_malformed_argument_is_refused() -> None:
    with pytest.raises(ValueError, match="NAME=value"):
        parse_settings(["CHAT_SOURCES"])


def test_one_factor_at_a_time_varies_each_setting_alone() -> None:
    points = build_points({"CHAT_SOURCES": [3, 8], "RERANK_ENABLED": [False]}, cross=False)

    assert points == (
        GridPoint(overrides={"CHAT_SOURCES": 3}),
        GridPoint(overrides={"CHAT_SOURCES": 8}),
        GridPoint(overrides={"RERANK_ENABLED": False}),
    )


def test_a_value_equal_to_baseline_is_dropped_not_rerun() -> None:
    baseline = config.CHAT_SOURCES

    points = build_points({"CHAT_SOURCES": [baseline, baseline + 1]}, cross=False)

    assert points == (GridPoint(overrides={"CHAT_SOURCES": baseline + 1}),)


def test_cross_takes_the_product_and_drops_the_all_baseline_point() -> None:
    baseline = config.CHAT_SOURCES

    points = build_points(
        {"CHAT_SOURCES": [baseline, 8], "RERANK_ENABLED": [config.RERANK_ENABLED, False]},
        cross=True,
    )

    assert points == (
        GridPoint(overrides={"RERANK_ENABLED": False}),
        GridPoint(overrides={"CHAT_SOURCES": 8}),
        GridPoint(overrides={"CHAT_SOURCES": 8, "RERANK_ENABLED": False}),
    )


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
    seen: list[int] = []

    async def fake_run(case):
        seen.append(config.CHAT_SOURCES)
        return eval_result(case=case)

    monkeypatch.setattr(service, "run_retrieval", fake_run)
    baseline = config.CHAT_SOURCES
    dataset = eval_dataset(eval_case(id="one"), eval_case(id="two"))
    points = (
        GridPoint(overrides={"CHAT_SOURCES": baseline + 1}),
        GridPoint(overrides={"CHAT_SOURCES": baseline + 2}),
    )

    run = await run_grid(dataset, points)

    assert seen == [baseline, baseline, baseline + 1, baseline + 1, baseline + 2, baseline + 2]
    assert config.CHAT_SOURCES == baseline
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
