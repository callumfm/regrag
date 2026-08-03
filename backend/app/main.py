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
from app.core.exceptions import (
    DomainError,
    domain_error_handler,
    http_exception_handler,
    integrity_error_handler,
    validation_error_handler,
)
from app.core.logger import setup_logging
from app.core.middleware import request_middleware
from app.db.session import async_engine

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle - teardown of shared resources."""
    try:
        yield
    finally:
        await async_engine.dispose()


def configure_app(app: FastAPI) -> None:
    """Register exception handlers, middleware, and routes.

    Kept separate from app construction so tests can wire a throwaway app
    identically to production."""
    # ty doesn't model Starlette's async handler variance; register via a loop
    # so we only need one suppression.
    exception_handlers = [
        (RequestValidationError, validation_error_handler),
        (HTTPException, http_exception_handler),
        (DomainError, domain_error_handler),
        (IntegrityError, integrity_error_handler),
    ]
    for exc_type, handler in exception_handlers:
        app.add_exception_handler(exc_type, handler)  # ty: ignore[invalid-argument-type]

    app.middleware("http")(request_middleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)


app = FastAPI(title=config.PROJECT_NAME, version=__version__, lifespan=lifespan)
configure_app(app)
