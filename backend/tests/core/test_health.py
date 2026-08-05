from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.db.session import get_db
from app.core.health import get_health
from app.ingestion.enums import IngestRunStatus
from app.ingestion.schemas import IngestRun


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "corpus_version": None,
        "database": "ok",
    }


def test_health_degraded_when_db_unreachable(app: FastAPI, client: TestClient) -> None:
    bad_engine = create_async_engine("postgresql+psycopg://postgres:postgres@localhost:9/regrag")
    bad_factory = async_sessionmaker(bind=bad_engine, class_=AsyncSession)

    async def bad_db() -> AsyncGenerator[AsyncSession, None]:
        async with bad_factory() as session:
            yield session

    app.dependency_overrides[get_db] = bad_db
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == "error"


@pytest.mark.anyio
async def test_health_reports_the_latest_corpus_version(db_session: AsyncSession) -> None:
    db_session.add(IngestRun(status=IngestRunStatus.COMPLETED, corpus_version="2026-08-04-abc1234"))
    await db_session.flush()

    response = await get_health(db_session)
    assert response.corpus_version == "2026-08-04-abc1234"
