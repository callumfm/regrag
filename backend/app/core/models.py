"""Shared Pydantic models."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class AppModel(BaseModel):
    """Base for every model: builds from ORM rows and Row objects as well as dicts."""

    model_config = ConfigDict(from_attributes=True)


class FrozenModel(AppModel):
    """Base for immutable, hashable domain values."""

    model_config = ConfigDict(frozen=True)


class ErrorResponse(AppModel):
    """The single JSON shape every error response uses."""

    error: str
    message: str
    request_id: str | None = None
    detail: list[Any] | None = None
