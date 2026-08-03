"""FastAPI application entrypoint."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes.health import router as health_router
from app.core.config import config
from app.core.exceptions import register_exception_handlers
from app.core.logger import setup_logging
from app.core.middleware import register_middleware
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
    """Wire the app identically for production and tests."""
    register_exception_handlers(app)
    register_middleware(app)
    app.include_router(health_router)


app = FastAPI(title=config.PROJECT_NAME, version=__version__, lifespan=lifespan)
configure_app(app)
