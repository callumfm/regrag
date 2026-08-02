"""Domain errors and exception handlers returning one consistent JSON shape."""

from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException

from app.core.context import request_id_var
from app.core.logger import get_logger
from app.core.models import ErrorResponse

logger = get_logger(__name__)


class DomainError(Exception):
    """Base for application errors that map to HTTP responses via one handler."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NotFoundError(DomainError):
    """Raised when a resource is not found."""

    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, resource: str, identifier: str | int):
        super().__init__(f"{resource} '{identifier}' not found")
        self.resource = resource
        self.identifier = identifier


class ConflictError(DomainError):
    """Raised when a request conflicts with the current state of a resource."""

    status_code = status.HTTP_409_CONFLICT


def require[T](obj: T | None, *, resource: str, identifier: str | int) -> T:
    """Return obj, raising NotFoundError if it is None."""
    if obj is None:
        raise NotFoundError(resource, identifier)
    return obj


def error_response(
    status_code: int,
    *,
    error: str,
    message: str,
    detail: list[Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build the single error shape every handler returns."""
    body = ErrorResponse(
        error=error,
        message=message,
        request_id=request_id_var.get(),
        detail=detail,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
        headers=headers,
    )


def _sanitize_validation_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip raw input values to avoid leaking request payloads into responses and logs."""
    return [{k: v for k, v in e.items() if k != "input"} for e in errors]


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors: 422 with sanitized error details."""
    sanitized = _sanitize_validation_errors(exc.errors())
    logger.error("Validation error on %s %s: %s", request.method, request.url.path, sanitized)
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        error="ValidationError",
        message="Request validation failed",
        detail=jsonable_encoder(sanitized),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions, preserving status and headers."""
    logger.warning(
        "HTTPException on %s %s: %s %s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return error_response(
        exc.status_code,
        error="HTTPException",
        message=str(exc.detail),
        headers=exc.headers,
    )


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """Map any DomainError subclass to a JSON response using its status_code."""
    name = type(exc).__name__
    logger.warning("%s on %s %s: %s", name, request.method, request.url.path, exc.message)
    return error_response(exc.status_code, error=name, message=exc.message)


async def integrity_error_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    """Handle database constraint violations as 409 Conflict."""
    logger.warning("IntegrityError on %s %s: %s", request.method, request.url.path, exc.orig)
    return error_response(
        status.HTTP_409_CONFLICT,
        error="IntegrityError",
        message="This conflicts with an existing record",
    )


async def catch_unhandled_exceptions(request: Request, call_next):
    """Convert unhandled exceptions to the shared 500 shape. Runs as innermost
    middleware (not an exception handler) so the request-ID contextvar is still bound."""
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="InternalServerError",
            message="An unexpected error occurred",
        )
