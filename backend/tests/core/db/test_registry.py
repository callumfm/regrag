"""Every ORM schema is registered so SQLAlchemy can configure its mappers."""

from sqlalchemy.orm import configure_mappers

import app.core.db.registry  # noqa: F401
from app.core.db.schema import BaseSchema


def test_registry_covers_the_ingest_tables():
    assert {"ingest_runs", "ingested_documents"} <= set(BaseSchema.metadata.tables)


def test_all_mappers_configure():
    """Fails if a relationship references a model missing from the registry."""
    configure_mappers()
