"""Grid construction: --set parsing, the curated default sweep, and point building."""

import pytest

from app.core.config import config
from app.evals.tune.grid import build_points, parse_param_values
from app.evals.tune.models import GridPoint
from app.evals.tune.params import TUNABLE_PARAMS


def test_set_arguments_parse_to_validated_values() -> None:
    parsed = parse_param_values(["CHAT_SOURCES=3,8", "RERANK_ENABLED=false"])

    assert parsed == {"CHAT_SOURCES": [3, 8], "RERANK_ENABLED": [False]}


def test_no_arguments_means_the_full_curated_sweep() -> None:
    parsed = parse_param_values([])

    assert set(parsed) == set(TUNABLE_PARAMS)
    assert parsed["CHAT_CONTEXT_CHUNKS"] == [10, 15, 20, 30]


def test_a_param_given_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="CHAT_SOURCES"):
        parse_param_values(["CHAT_SOURCES=3", "CHAT_SOURCES=8"])


def test_a_malformed_argument_is_refused() -> None:
    with pytest.raises(ValueError, match="NAME=value"):
        parse_param_values(["CHAT_SOURCES"])


def test_one_factor_at_a_time_varies_each_param_alone() -> None:
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
