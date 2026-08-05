"""Health endpoint and its response model."""

from enum import StrEnum

from fastapi import APIRouter
from pydantic import BaseModel, computed_field
from sqlalchemy import text

from app import __version__
from app.core.dependencies import SessionDep
from app.ingestion.service import get_latest_corpus_version


class ServiceStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    version: str = __version__
    corpus_version: str | None = None
    database: ServiceStatus

    @computed_field
    @property
    def status(self) -> HealthStatus:
        """Overall status: ok only while every ServiceStatus field reports ok."""
        services = (
            getattr(self, name)
            for name, field in type(self).model_fields.items()
            if field.annotation is ServiceStatus
        )
        if all(service is ServiceStatus.OK for service in services):
            return HealthStatus.OK
        return HealthStatus.DEGRADED


router = APIRouter(tags=["health"])


@router.get("/health")
async def get_health(db: SessionDep) -> HealthResponse:
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        return HealthResponse(database=ServiceStatus.ERROR)
    try:
        corpus_version = await get_latest_corpus_version(db)
    except Exception:
        corpus_version = None
    return HealthResponse(database=ServiceStatus.OK, corpus_version=corpus_version)
