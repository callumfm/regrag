"""Application-wide enumerations."""

from enum import StrEnum


class Environment(StrEnum):
    """Application environment."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class IngestRunStatus(StrEnum):
    """Lifecycle of a corpus ingest run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
