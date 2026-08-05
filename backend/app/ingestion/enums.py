"""Ingestion enumerations."""

from enum import StrEnum


class IngestRunStatus(StrEnum):
    """Lifecycle of a corpus ingest run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DocAction(StrEnum):
    """How a discovered document compares to the previous run."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class SectionKind(StrEnum):
    """Node types in a parsed document tree."""

    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    ANNEX = "annex"
    HEADING = "heading"
    TABLE = "table"
