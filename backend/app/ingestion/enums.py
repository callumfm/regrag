"""Ingestion enumerations."""

from enum import StrEnum


class IngestRunStatus(StrEnum):
    """Lifecycle of a corpus ingest run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DocChange(StrEnum):
    """How a discovered document compares to the previous run."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"

    @classmethod
    def between(cls, previous: str | None, current: str) -> "DocChange":
        """How this run's resolved celex compares to the previous run's, if there was one."""
        if previous is None:
            return cls.NEW
        return cls.CHANGED if previous != current else cls.UNCHANGED


class SectionKind(StrEnum):
    """Node types in a parsed document tree."""

    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    ANNEX = "annex"
    HEADING = "heading"
    TABLE = "table"
