from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.session import get_db


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
    bad_engine = create_async_engine("postgresql+asyncpg://postgres:postgres@localhost:9/regrag")
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
