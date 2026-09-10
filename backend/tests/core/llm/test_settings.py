"""What a role's model call is made with, the parameters a provider is spared, and the boot
check that its key is present."""

import litellm
import pytest
from litellm.utils import get_optional_params

import app.core.llm  # noqa: F401 — importing the package is what sets litellm's globals
from app.core.llm.settings import MissingModelKeyError, ModelSettings, check_model_keys


def test_a_role_carries_its_own_model_and_limits():
    settings = ModelSettings(model="openai/gpt-5", timeout=30, max_tokens=512)

    assert settings.model == "openai/gpt-5"
    assert settings.timeout == 30
    assert settings.max_tokens == 512


def test_a_role_that_does_not_configure_sampling_sends_no_temperature():
    """litellm omits a None, so the judge keeps the provider's own default rather than
    silently acquiring the answer's temperature and re-scoring every past run."""
    settings = ModelSettings(model="anthropic/claude-sonnet-5", timeout=60, max_tokens=1)

    assert settings.temperature is None


def test_a_parameter_the_model_refuses_is_dropped_rather_than_raised():
    """The whole point of the global: without it, temperature to a frontier Anthropic model
    raises UnsupportedParamsError, so pointing a role there would be a code change."""
    assert litellm.drop_params is True

    sent = get_optional_params(
        model="claude-sonnet-5", custom_llm_provider="anthropic", temperature=0.0, max_tokens=10
    )

    assert sent == {"max_tokens": 10}


def test_a_parameter_the_model_accepts_is_still_sent():
    sent = get_optional_params(
        model="claude-haiku-4-5", custom_llm_provider="anthropic", temperature=0.0, max_tokens=10
    )

    assert sent == {"temperature": 0.0, "max_tokens": 10}


def test_a_role_leaving_temperature_unset_sends_nothing_rather_than_null():
    sent = get_optional_params(
        model="claude-sonnet-5", custom_llm_provider="anthropic", temperature=None, max_tokens=10
    )

    assert "temperature" not in sent


def test_a_configured_key_passes_the_check(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")

    check_model_keys("anthropic/claude-haiku-4-5")


def test_a_missing_key_names_the_model_and_the_variable(monkeypatch):
    """The failure has to say which of several configured models is unusable, and which
    variable to set, or a provider swap debugs as a 401 on the first request."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingModelKeyError, match="openai/gpt-5.*OPENAI_API_KEY"):
        check_model_keys("openai/gpt-5")


def test_the_check_reports_every_unusable_model_at_once(monkeypatch):
    """One boot, one fix: a run missing two providers' keys should not need two attempts."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")

    with pytest.raises(MissingModelKeyError) as failure:
        check_model_keys("openai/gpt-5", "anthropic/claude-haiku-4-5", "gemini/gemini-2.5-pro")

    assert "openai/gpt-5" in str(failure.value)
    assert "gemini/gemini-2.5-pro" in str(failure.value)
    assert "claude-haiku" not in str(failure.value)


def test_a_model_named_twice_is_checked_once(monkeypatch):
    """Four chat roles share one default model; the failure should name it once."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingModelKeyError) as failure:
        check_model_keys("openai/gpt-5", "openai/gpt-5")

    assert str(failure.value).count("openai/gpt-5") == 1
