from enum import StrEnum

from pydantic import BaseModel, computed_field

from app import __version__


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
