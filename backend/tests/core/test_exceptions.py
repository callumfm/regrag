"""Tests for exception handlers: one JSON shape everywhere."""

from typing import Any

import pytest
from fastapi import APIRouter, HTTPException
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import ConflictError, NotFoundError, require
from app.main import create_app


class _Payload(BaseModel):
    name: str
    count: int


def _build_client() -> TestClient:
    app = create_app()
    router = APIRouter()

    @router.get("/boom-domain")
    def boom_domain() -> None:
        raise NotFoundError("Regulation", "fueleu")

    @router.get("/boom-conflict")
    def boom_conflict() -> None:
        raise ConflictError("Chunk already embedded")

    @router.get("/boom-http")
    def boom_http() -> None:
        raise HTTPException(status_code=418, detail="teapot", headers={"X-Tea": "yes"})

    @router.get("/boom-integrity")
    def boom_integrity() -> None:
        raise IntegrityError("INSERT ...", {}, Exception("duplicate key value"))

    @router.get("/boom-unhandled")
    def boom_unhandled() -> None:
        raise RuntimeError("secret internal detail")

    @router.post("/echo")
    def echo(payload: _Payload) -> _Payload:
        return payload

    app.include_router(router)
    return TestClient(app)


client = _build_client()


def assert_error_shape(response: Response, status_code: int, error: str) -> dict[str, Any]:
    """Assert the single error schema: {error, message, request_id} + optional detail."""
    assert response.status_code == status_code
    body = response.json()
    assert body["error"] == error
    assert isinstance(body["message"], str) and body["message"]
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert set(body) <= {"error", "message", "request_id", "detail"}
    return body


def test_domain_error() -> None:
    body = assert_error_shape(client.get("/boom-domain"), 404, "NotFoundError")
    assert body["message"] == "Regulation 'fueleu' not found"


def test_conflict_error() -> None:
    body = assert_error_shape(client.get("/boom-conflict"), 409, "ConflictError")
    assert body["message"] == "Chunk already embedded"


def test_http_exception_preserves_headers() -> None:
    response = client.get("/boom-http")
    assert_error_shape(response, 418, "HTTPException")
    assert response.headers["X-Tea"] == "yes"


def test_integrity_error_hides_db_detail() -> None:
    body = assert_error_shape(client.get("/boom-integrity"), 409, "IntegrityError")
    assert "duplicate key value" not in body["message"]


def test_unhandled_error_leaks_nothing() -> None:
    response = client.get("/boom-unhandled")
    assert_error_shape(response, 500, "InternalServerError")
    assert "secret internal detail" not in response.text


def test_unhandled_error_still_has_process_time() -> None:
    response = client.get("/boom-unhandled")
    assert "X-Process-Time" in response.headers


def test_validation_error_strips_input() -> None:
    response = client.post("/echo", json={"name": "x", "count": "not-a-number"})
    body = assert_error_shape(response, 422, "ValidationError")
    assert body["detail"]
    assert all("input" not in item for item in body["detail"])


def test_unknown_route_uses_shared_schema() -> None:
    assert_error_shape(client.get("/nope"), 404, "HTTPException")


def test_require_returns_value() -> None:
    assert require("x", resource="Thing", identifier=1) == "x"


def test_require_raises_not_found() -> None:
    with pytest.raises(NotFoundError):
        require(None, resource="Thing", identifier=1)
