"""Application-wide enumerations."""

from enum import StrEnum


class Environment(StrEnum):
    """Application environment."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"
