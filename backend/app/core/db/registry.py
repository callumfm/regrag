"""Single import point for every ORM schema so all mappers register.

Add new capability schemas here; the guard test fails if one is missing.
"""

from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun

__all__ = ["DocumentChunk", "IngestRun", "RawDocument"]
