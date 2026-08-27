"""Grid tuning: parsing, point construction, and the config-override run loop."""

import pytest
from pydantic import ValidationError

from app.core.config import config
from app.evals.tune.models import GridPoint
from app.evals.tune.service import build_points, parse_settings, tunable_fields


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
