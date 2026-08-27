"""The curated params: valid against the live config, coerced by their own fields."""

import pytest
from pydantic import ValidationError

from app.core.config import Config
from app.evals.tune.params import TUNABLE_PARAMS, get_tunable_params, validate_value


def test_every_curated_param_and_value_still_fits_the_config() -> None:
    """The drift gate: a renamed or retightened setting fails here, not mid-sweep."""
    params = get_tunable_params()

    assert set(params) == set(TUNABLE_PARAMS)
    assert all(name in Config.model_fields for name in params)


def test_values_are_coerced_by_their_field_types() -> None:
    assert validate_value("CHAT_SOURCES", "3") == 3
    assert validate_value("RERANK_ENABLED", "false") is False


def test_an_out_of_range_value_fails_before_any_run() -> None:
    with pytest.raises(ValidationError):
        validate_value("CHAT_SOURCES", 0)


def test_an_unknown_param_is_named_with_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="RERANK_TIMEOUT is not a tunable param"):
        validate_value("RERANK_TIMEOUT", 60)
