from fastapi import APIRouter
from sqlalchemy import text

from app.api.deps import SessionDep
from app.api.models.health import HealthResponse, ServiceStatus

router = APIRouter(tags=["health"])


@router.get("/health")
async def get_health(db: SessionDep) -> HealthResponse:
    try:
        await db.execute(text("SELECT 1"))
        database = ServiceStatus.OK
    except Exception:
        database = ServiceStatus.ERROR
    return HealthResponse(database=database)
