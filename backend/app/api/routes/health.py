from typing import Literal

from fastapi import APIRouter
from sqlalchemy import text

from app import __version__
from app.api.deps import SessionDep
from app.api.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def get_health(db: SessionDep) -> HealthResponse:
    database: Literal["ok", "error"]
    try:
        await db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "error"
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        version=__version__,
        corpus_version=None,
        database=database,
    )
