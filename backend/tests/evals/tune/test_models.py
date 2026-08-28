"""Tune values: the curated-param model's drift gate and its override window."""

import pytest
from pydantic import ValidationError

from app.core.config import config
from app.evals.tune.models import TunableParam


def test_a_param_validates_its_values_against_the_live_config() -> None:
    TunableParam(name="CHAT_SOURCES", values=(3, 8)).validate_config()


def test_a_renamed_setting_fails_the_drift_gate() -> None:
    with pytest.raises(ValueError, match="no longer a config field"):
        TunableParam(name="RENAMED_AWAY", values=(1,)).validate_config()


def test_an_out_of_range_value_fails_before_any_run() -> None:
    with pytest.raises(ValidationError):
        TunableParam(name="CHAT_SOURCES", values=(0,)).validate_config()


def test_validation_never_touches_the_live_config() -> None:
    baseline = config.CHAT_SOURCES

    TunableParam(name="CHAT_SOURCES", values=(baseline + 3,)).validate_config()

    assert config.CHAT_SOURCES == baseline


def test_override_applies_and_restores_the_setting() -> None:
    baseline = config.CHAT_SOURCES
    param = TunableParam(name="CHAT_SOURCES", values=(baseline + 3,))

    with param.override(baseline + 3):
        assert config.CHAT_SOURCES == baseline + 3

    assert config.CHAT_SOURCES == baseline


def test_override_restores_even_when_the_block_raises() -> None:
    baseline = config.CHAT_SOURCES
    param = TunableParam(name="CHAT_SOURCES", values=(baseline + 3,))

    with pytest.raises(RuntimeError):
        with param.override(baseline + 3):
            raise RuntimeError("boom")

    assert config.CHAT_SOURCES == baseline
