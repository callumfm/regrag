"""Async SQLAlchemy database engine and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import config

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
    """Get a database session, rollback on error or interrupt and ensure session is closed."""
    async with async_session_factory() as db:
        try:
            yield db
            if auto_commit:
                await db.commit()
        except BaseException:
            await db.rollback()
            raise
