"""Tests for request-ID and process-time middleware."""

import re
import uuid

from fastapi.testclient import TestClient

from app.core import middleware
from app.main import create_app

client = TestClient(create_app())


def test_every_response_carries_request_id() -> None:
    response = client.get("/health")
    uuid.UUID(hex=response.headers["X-Request-ID"])


def test_requests_get_distinct_ids() -> None:
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers["X-Request-ID"] != r2.headers["X-Request-ID"]


def test_incoming_request_id_is_ignored() -> None:
    incoming = "attacker-chosen-value"
    response = client.get("/health", headers={"X-Request-ID": incoming})
    assert response.headers["X-Request-ID"] != incoming


def test_process_time_header() -> None:
    response = client.get("/health")
    assert re.fullmatch(r"\d+ms", response.headers["X-Process-Time"])


def test_access_log_line(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(middleware.logger, "info", lambda msg, *a, **k: calls.append(msg))
    client.get("/health")
    assert len(calls) == 1
    assert "GET /health 200" in calls[0]


def test_access_log_skips_options(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(middleware.logger, "info", lambda msg, *a, **k: calls.append(msg))
    client.options("/health")
    assert calls == []


def test_gzip_compresses_large_responses() -> None:
    app = create_app()

    @app.get("/big")
    def big() -> dict[str, str]:
        return {"payload": "x" * 5000}

    response = TestClient(app).get("/big", headers={"Accept-Encoding": "gzip"})
    assert response.headers["Content-Encoding"] == "gzip"
    assert response.json()["payload"] == "x" * 5000
