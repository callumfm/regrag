"""Generic persistence helpers shared by the per-domain service modules."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.schema import BaseSchema


async def create_record[T: BaseSchema](
    session: AsyncSession, record: T, *, commit: bool = True
) -> T:
    """Persist a new record; commits by default so it outlives the caller's transaction."""
    session.add(record)
    await (session.commit() if commit else session.flush())
    return record


async def update_record[T: BaseSchema](
    session: AsyncSession,
    record: T,
    updates: BaseModel | Mapping[str, Any],
    *,
    commit: bool = True,
) -> T:
    """Apply field updates to a record; Pydantic models contribute only their set fields."""
    if isinstance(updates, BaseModel):
        updates = updates.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(record, field, value)
    if commit:
        await session.commit()
    return record
