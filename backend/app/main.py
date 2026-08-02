"""FastAPI application entrypoint."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException

from app import __version__
from app.api.routes.health import router as health_router
from app.core.config import config
from app.core.constants import GZIP_MINIMUM_SIZE
from app.core.exceptions import (
    DomainError,
    catch_unhandled_exceptions,
    domain_error_handler,
    http_exception_handler,
    integrity_error_handler,
    validation_error_handler,
)
from app.core.middleware import add_process_time_header, request_id_middleware
from app.db.session import async_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle - teardown of shared resources."""
    try:
        yield
    finally:
        await async_engine.dispose()


app = FastAPI(title=config.PROJECT_NAME, version=__version__, lifespan=lifespan)

# Exception Handlers — ty doesn't model Starlette's async handler variance;
# register via a loop so we only need one suppression.
exception_handlers = [
    (RequestValidationError, validation_error_handler),
    (HTTPException, http_exception_handler),
    (DomainError, domain_error_handler),
    (IntegrityError, integrity_error_handler),
]
for exc_type, handler in exception_handlers:
    app.add_exception_handler(exc_type, handler)  # ty: ignore[invalid-argument-type]

# Middleware — later registrations wrap earlier ones; catch_unhandled_exceptions
# must stay innermost so 500s still carry the request ID.
app.middleware("http")(catch_unhandled_exceptions)
app.middleware("http")(add_process_time_header)
app.middleware("http")(request_id_middleware)
app.add_middleware(GZipMiddleware, minimum_size=GZIP_MINIMUM_SIZE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
