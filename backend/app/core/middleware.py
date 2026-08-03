"""HTTP middleware for the FastAPI application."""

import logging
import time
import uuid

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.config import config
from app.core.exceptions import error_response
from app.core.logger import request_id_var

logger = logging.getLogger(__name__)


async def request_id_middleware(request: Request, call_next):
    """Bind a server-generated request ID for the request's lifetime."""
    request_id = uuid.uuid4().hex
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


async def process_time_middleware(request: Request, call_next):
    """Time the request and emit one access-log line."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Process-Time"] = f"{duration_ms}ms"
    logger.info(
        "%s %s %s - %sms",
        request.method,
        request.scope["path"],
        response.status_code,
        duration_ms,
        extra={
            "method": request.method,
            "path": request.scope["path"],
            "route": getattr(request.scope.get("route"), "path", None),
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else None,
        },
    )
    return response


async def exception_middleware(request: Request, call_next):
    """Convert unhandled exceptions into the shared 500 error shape."""
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.scope["path"])
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="InternalServerError",
            message="An unexpected error occurred",
        )


def register_middleware(app: FastAPI) -> None:
    """Register middleware, innermost first (Starlette runs the last-added
    middleware first): exception sits innermost so the access log records the
    500, request-ID outermost of the three so the contextvar is set for all
    downstream logging."""
    app.middleware("http")(exception_middleware)
    app.middleware("http")(process_time_middleware)
    app.middleware("http")(request_id_middleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
