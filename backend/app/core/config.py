"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from typing import Any

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.enums import Environment

ENVIRONMENT: Environment = Environment(os.environ.get("ENVIRONMENT", "dev"))
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_env_file(env: Environment = ENVIRONMENT) -> str:
    if env == Environment.TEST:
        return ".env.example"
    return f".env.{env.value}"


MODEL_CONFIG = SettingsConfigDict(
    env_file=get_env_file(),
    env_ignore_empty=True,
    extra="ignore",
)


class AppConfig(BaseSettings):
    """Application configuration."""

    model_config = MODEL_CONFIG

    ENVIRONMENT: Environment = ENVIRONMENT
    PROJECT_NAME: str = "RegRag"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"


class PostgresConfig(BaseSettings):
    """PostgreSQL database configuration."""

    model_config = MODEL_CONFIG

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASS: str = "postgres"
    DB_NAME: str = "regrag"

    DB_POOL_SIZE: int = 3
    DB_MAX_OVERFLOW: int = 3
    DB_POOL_PRE_PING: bool = True
    DB_POOL_RECYCLE: int = 300
    DB_POOL_TIMEOUT: int = 30
    DB_CONNECT_TIMEOUT: int = 10
    DB_COMMAND_TIMEOUT: int = 30

    @computed_field
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Build SQLAlchemy database URI."""
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @computed_field
    @property
    def SQLALCHEMY_ENGINE_ARGS(self) -> dict[str, Any]:
        """Get SQLAlchemy engine arguments."""
        return {
            "pool_size": self.DB_POOL_SIZE,
            "max_overflow": self.DB_MAX_OVERFLOW,
            "pool_pre_ping": self.DB_POOL_PRE_PING,
            "pool_recycle": self.DB_POOL_RECYCLE,
            "pool_timeout": self.DB_POOL_TIMEOUT,
            "connect_args": {
                "connect_timeout": self.DB_CONNECT_TIMEOUT,
                "options": f"-c statement_timeout={self.DB_COMMAND_TIMEOUT * 1000}",
            },
        }


class Config(AppConfig, PostgresConfig):
    """Combined configuration class for core app functionality."""


def load_config() -> Config:
    """Load the configuration."""
    return Config()


config = load_config()
