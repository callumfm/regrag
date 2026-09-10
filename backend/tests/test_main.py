"""Tests for application lifespan."""

from fastapi.testclient import TestClient

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
