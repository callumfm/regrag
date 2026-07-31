from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    corpus_version: str | None


@router.get("/health")
def get_health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, corpus_version=None)
