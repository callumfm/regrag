"""The parameters a provider is spared, and the boot check that its key is present."""

import litellm
import pytest
from litellm.utils import get_optional_params

import app.core.llm  # noqa: F401 — importing the package is what sets litellm's globals
from app.core.config import config, configured_models
from app.core.llm.keys import MissingModelKeyError, check_model_keys


def test_a_parameter_the_model_refuses_is_dropped_rather_than_raised():
    """Without the global, temperature to a frontier Anthropic model raises
    UnsupportedParamsError, so pointing a role there would be a code change."""
    assert litellm.drop_params is True

    sent = get_optional_params(
        model="claude-sonnet-5", custom_llm_provider="anthropic", temperature=0.0, max_tokens=10
    )

    assert sent == {"max_tokens": 10}


def test_every_model_setting_is_checked_without_naming_one(monkeypatch):
    """Read off the settings whose name says model, so a role added later is covered."""
    monkeypatch.setattr(config, "CHAT_MODEL", "openai/gpt-5")

    assert set(configured_models()) == {
        "openai/gpt-5",
        config.EMBED_MODEL,
        config.RERANK_MODEL,
        config.EVAL_JUDGE_MODEL,
    }


def test_a_model_shared_by_two_settings_is_named_once():
    """The chat model and the judge may name the same model; the check should not ask
    after its key twice."""
    assert len(configured_models()) == len(set(configured_models()))


def test_a_configured_key_passes_the_check(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")

    check_model_keys()


def test_a_missing_key_names_the_variable_and_what_wants_it(monkeypatch):
    """The failure has to say which variable to set and which model wants it, or a provider
    swap debugs as a 401 on the first request."""
    monkeypatch.setattr(config, "CHAT_MODEL", "openai/gpt-5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingModelKeyError, match="OPENAI_API_KEY for openai/gpt-5"):
        check_model_keys()


def test_the_check_reports_every_unset_variable_at_once(monkeypatch):
    """One boot, one fix: a deployment missing two providers' keys should not need two
    attempts to learn that."""
    monkeypatch.setattr(config, "CHAT_MODEL", "openai/gpt-5")
    monkeypatch.setattr(config, "EVAL_JUDGE_MODEL", "gemini/gemini-2.5-pro")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(MissingModelKeyError) as failure:
        check_model_keys()

    assert "OPENAI_API_KEY" in str(failure.value)
    assert "GEMINI_API_KEY" in str(failure.value)


def test_one_variable_serving_two_models_is_asked_for_once(monkeypatch):
    """Embed and rerank are both Voyage; the fix is one variable, so the failure should say
    it once and name both models under it."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    with pytest.raises(MissingModelKeyError) as failure:
        check_model_keys()

    assert str(failure.value).count("VOYAGE_API_KEY") == 1
    assert config.EMBED_MODEL in str(failure.value)
    assert config.RERANK_MODEL in str(failure.value)
