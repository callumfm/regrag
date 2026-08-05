"""Shared Pydantic models."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Base for immutable, hashable domain values."""

    model_config = ConfigDict(frozen=True)


class ErrorResponse(BaseModel):
    """The single JSON shape every error response uses."""

    error: str
    message: str
    request_id: str | None = None
    detail: list[Any] | None = None
