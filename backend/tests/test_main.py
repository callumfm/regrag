"""Tests for application lifespan."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import config
from app.core.llm.settings import MissingModelKeyError
from app.main import app


def test_lifespan_disposes_engine(monkeypatch) -> None:
    calls: list[bool] = []

    class FakeEngine:
        async def dispose(self) -> None:
            calls.append(True)

    monkeypatch.setattr("app.main.async_engine", FakeEngine())
    with TestClient(app):
        pass
    assert calls == [True]


def test_the_api_refuses_to_boot_when_a_model_it_may_reach_has_no_key(monkeypatch) -> None:
    """A key missing at boot beats a 502 on the first question a user asks."""
    monkeypatch.setattr(config, "ASSESS_MODEL", "openai/gpt-5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingModelKeyError, match="openai/gpt-5.*OPENAI_API_KEY"):
        with TestClient(app):
            pass


def test_the_boot_check_spares_the_judge(monkeypatch) -> None:
    """Only an eval run grades an answer, so serving chat must not need the judge's key."""
    monkeypatch.setattr(config, "EVAL_JUDGE_MODEL", "openai/gpt-5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with TestClient(app):
        pass
