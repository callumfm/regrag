"""Shared test fixtures."""

from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from tenacity import wait_none

from app.core.config import config
from app.core.db.session import async_session_factory
from app.ingestion.discover import discover
from app.ingestion.eurlex import resolve
from app.ingestion.fetch import download
from app.ingestion.schemas import IngestedDocument, IngestRun
from app.main import configure_app

RETRIED = (discover, resolve, download)


@pytest.fixture
def defuse_retry(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable], Callable]:
    """Strip tenacity's waits from a @transient_retry function, keeping its retry behaviour."""

    def _defuse(fn: Callable) -> Callable:
        # ty: ignore[unresolved-attribute] — tenacity sets .retry dynamically, untyped
        monkeypatch.setattr(fn.retry, "wait", wait_none())
        return fn

    return _defuse


@pytest.fixture(autouse=True)
def no_retry_backoff(defuse_retry: Callable[[Callable], Callable]) -> None:
    """Defuse every @transient_retry callable in RETRIED, so retry tests don't sleep."""
    for fn in RETRIED:
        defuse_retry(fn)


@pytest.fixture(scope="session")
def db_engine() -> AsyncEngine:
    """NullPool so each test's connection lives and dies inside its own event loop."""
    return create_async_engine(config.SQLALCHEMY_DATABASE_URI, poolclass=NullPool)


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Session bound to a transaction that is always rolled back, with ingest tables cleared."""
    async with db_engine.connect() as conn:
        trans = await conn.begin()
        await conn.execute(delete(IngestedDocument))
        await conn.execute(delete(IngestRun))
        async with async_session_factory(bind=conn) as session:
            yield session
        await trans.rollback()


@pytest.fixture
def make_document() -> Callable[..., IngestedDocument]:
    """Build an IngestedDocument whose identity fields derive from ref, overridable per field."""

    def _make(run: IngestRun, ref: str = "32023R1805", **overrides: Any) -> IngestedDocument:
        defaults: dict[str, Any] = {
            "run": run,
            "name": ref,
            "source": "eurlex",
            "ref": ref,
            "resolved_ref": ref,
            "topic": "fueleu",
            "url": f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{ref}",
            "sha256": "a" * 64,
            "size_bytes": 758462,
            "fetched_at": datetime.now(UTC),
        }
        return IngestedDocument(**{**defaults, **overrides})

    return _make


@pytest.fixture
def app() -> FastAPI:
    """A throwaway app wired like production, so tests never mutate the real one."""
    app = FastAPI()
    configure_app(app)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)
