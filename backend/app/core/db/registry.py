"""Single import point for every ORM schema so all mappers register.

Add new capability schemas here; the guard test fails if one is missing.
"""

from app.ingestion.schemas import IngestedDocument, IngestRun

__all__ = ["IngestRun", "IngestedDocument"]
