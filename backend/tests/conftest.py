"""Shared test fixtures."""

from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from tenacity import wait_none

from app.core.config import config
from app.db.schemas import IngestedDocument, IngestRun
from app.db.session import async_session_factory
from app.ingestion.discover import discover
from app.ingestion.eurlex import resolve
from app.ingestion.fetch import download
from app.main import configure_app

RETRIED = (discover, resolve, download)


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tenacity's retry behaviour but drop its waits, so retry tests don't sleep."""
    for fn in RETRIED:
        # ty: ignore[unresolved-attribute] — tenacity sets .retry dynamically, untyped
        monkeypatch.setattr(fn.retry, "wait", wait_none())


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
def app() -> FastAPI:
    """A throwaway app wired like production, so tests never mutate the real one."""
    app = FastAPI()
    configure_app(app)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)
