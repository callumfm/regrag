"""Async SQLAlchemy database engine and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import config


class BaseSchema(DeclarativeBase):
    """Base for database schema with automatic created_at and updated_at timestamps.

    Alembic autogenerate targets its metadata.
    """

    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        server_default=func.now(),
        onupdate=func.now(),
    )


async_engine = create_async_engine(
    config.SQLALCHEMY_DATABASE_URI,
    **config.SQLALCHEMY_ENGINE_ARGS,
)

async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


@asynccontextmanager
async def get_session(*, auto_commit: bool = True) -> AsyncGenerator[AsyncSession, None]:
    """Get a database session, rollback on error and ensure session is closed."""
    async with async_session_factory() as db:
        try:
            yield db
            if auto_commit:
                await db.commit()
        except Exception:
            await db.rollback()
            raise
