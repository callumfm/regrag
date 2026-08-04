"""Declarative base for database schemas."""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class BaseSchema(DeclarativeBase):
    """Base for database schema with automatic created_at and updated_at timestamps.

    Alembic autogenerate targets its metadata. Timestamps are timezone-aware and
    enums are stored as their string values, without native Postgres enum types.
    Server-side defaults come back via RETURNING, so writes need no follow-up SELECT.
    """

    __mapper_args__ = {"eager_defaults": True}

    type_annotation_map = {
        datetime: DateTime(timezone=True),
        enum.Enum: Enum(
            enum.Enum, native_enum=False, values_callable=lambda e: [m.value for m in e]
        ),
    }

    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        server_default=func.now(),
        onupdate=func.now(),
    )
