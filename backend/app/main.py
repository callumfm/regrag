from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app import __version__
from app.api.routes.health import router as health_router
from app.core.config import config
from app.core.constants import GZIP_MINIMUM_SIZE
from app.core.middleware import add_process_time_header, request_id_middleware


def create_app() -> FastAPI:
    app = FastAPI(title=config.PROJECT_NAME, version=__version__)

    # Middleware. Registration order matters: later registrations wrap earlier
    # ones, so request-ID is outermost of the two functions and the process-time
    # log line runs with the request ID already bound.
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
    return app


app = create_app()
