"""HTTP middleware for the FastAPI application."""

import time
import uuid

from fastapi import Request

from app.core.context import request_id_var
from app.core.logger import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


async def request_id_middleware(request: Request, call_next):
    """Bind a server-generated request ID to the request context.
    Incoming X-Request-ID headers are untrusted (public API) and ignored."""
    request_id = uuid.uuid4().hex
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        request_id_var.reset(token)


async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header and log one access-log line per request."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    response.headers["X-Process-Time"] = f"{duration_ms}ms"
    if request.method != "OPTIONS":
        route = request.scope.get("route")
        logger.info(
            f"{request.method} {request.url.path} {response.status_code} - {duration_ms}ms",
            extra={
                "method": request.method,
                "path": request.url.path,
                "route": getattr(route, "path", None),
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client_ip": request.client.host if request.client else None,
            },
        )
    return response
