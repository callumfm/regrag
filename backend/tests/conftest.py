"""Shared test fixtures."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import configure_app


@pytest.fixture
def app() -> FastAPI:
    """A throwaway app wired like production, so tests never mutate the real one."""
    app = FastAPI()
    configure_app(app)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)
