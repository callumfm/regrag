from fastapi import APIRouter

from app import __version__
from app.api.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, corpus_version=None)
