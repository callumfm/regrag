"""HTTP middleware for the FastAPI application."""

import logging
import time
import uuid

from fastapi import Request, status

from app.core.exceptions import error_response
from app.core.logger import request_id_var

logger = logging.getLogger(__name__)


async def request_middleware(request: Request, call_next):
    """Bind a server-generated request ID (incoming X-Request-ID headers are
    untrusted and ignored), time the request, emit one access-log line, and
    convert unhandled exceptions into the shared 500 error shape.

    CORS preflights never reach this middleware — CORSMiddleware sits outside
    it and answers them directly — so they are absent from the access log."""
    request_id = uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start_time = time.perf_counter()
    method, path = request.method, request.scope["path"]
    try:
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error on %s %s", method, path)
            response = error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                error="InternalServerError",
                message="An unexpected error occurred",
            )
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms}ms"
        logger.info(
            "%s %s %s - %sms",
            method,
            path,
            response.status_code,
            duration_ms,
            extra={
                "method": method,
                "path": path,
                "route": getattr(request.scope.get("route"), "path", None),
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else None,
            },
        )
        return response
    finally:
        request_id_var.reset(token)
