"""Generic persistence helpers shared by the per-domain service modules."""

from collections.abc import Mapping
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.schemas.base import BaseSchema


async def create_record[T: BaseSchema](
    session: AsyncSession, record: T, *, commit: bool = True
) -> T:
    """Persist a new record; commits by default so server-side defaults are populated."""
    session.add(record)
    await (session.commit() if commit else session.flush())
    return record


async def update_record[T: BaseSchema](
    session: AsyncSession, record: T, updates: Mapping[str, Any], *, commit: bool = True
) -> T:
    """Apply field updates to a record, committing by default."""
    for field, value in updates.items():
        setattr(record, field, value)
    if commit:
        await session.commit()
    return record
