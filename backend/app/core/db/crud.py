"""Generic persistence helpers shared by the per-domain service modules."""

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.schema import BaseSchema


async def create_record[T: BaseSchema](session: AsyncSession, record: T) -> T:
    """Persist a new record, committed so it outlives the caller's transaction."""
    session.add(record)
    await session.commit()
    return record


async def update_record[T: BaseSchema](session: AsyncSession, record: T, updates: BaseModel) -> T:
    """Apply a partial update to a record; only the update model's set fields contribute."""
    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(record, field, value)
    await session.commit()
    return record
