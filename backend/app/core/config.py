"""Application configuration loaded from environment variables."""

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Application environment."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


def load_environment() -> Environment:
    """Read $ENVIRONMENT, naming the accepted values when it holds something else."""
    name = os.environ.get("ENVIRONMENT", Environment.DEV)
    try:
        return Environment(name)
    except ValueError:
        accepted = ", ".join(Environment)
        raise ValueError(
            f"ENVIRONMENT={name!r} is not a valid environment; expected one of: {accepted}"
        ) from None


ENVIRONMENT: Environment = load_environment()
BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


def get_env_file(env: Environment = ENVIRONMENT) -> Path:
    """Absolute, so the file is found whatever directory the process was started from."""
    name = ".env.example" if env == Environment.TEST else f".env.{env.value}"
    return BACKEND_ROOT / name


class BaseConfig(BaseSettings):
    """Base for the per-concern settings classes."""

    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_ignore_empty=True,
        extra="ignore",
    )


class AppConfig(BaseConfig):
    """Application configuration."""

    ENVIRONMENT: Environment = ENVIRONMENT
    PROJECT_NAME: str = "RegRag"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]


class StorageBackend(StrEnum):
    """Where raw source documents are kept."""

    LOCAL = "local"
    R2 = "r2"


class StorageConfig(BaseConfig):
    """Which backend holds raw source documents, and where the local one keeps them."""

    STORAGE_BACKEND: StorageBackend = StorageBackend.LOCAL
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"


class R2Config(BaseConfig):
    """Cloudflare R2 credentials, every field required so a half-configured bucket fails early."""

    R2_ACCOUNT_ID: str
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_BUCKET: str

    @property
    def R2_ENDPOINT_URL(self) -> str:
        """The S3-compatible endpoint R2 serves this account's buckets on."""
        return f"https://{self.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"


class PostgresConfig(BaseConfig):
    """PostgreSQL database configuration."""

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

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Build SQLAlchemy database URI."""
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

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


EMBED_DIMENSIONS = 1024
"""Width of the document_chunks.embedding column: not a setting, changing it needs a migration."""


class EmbeddingConfig(BaseConfig):
    """Voyage embedding configuration."""

    VOYAGE_API_KEY: str = ""
    EMBED_MODEL: str = "voyage/voyage-4-lite"
    EMBED_TIMEOUT: int = 30


class ChatConfig(BaseConfig):
    """Chat model configuration for the chat graph.

    CHAT_SOURCES: search hits the answer draws on, each widened to its section.
    CHAT_CONTEXT_CHUNKS: the most chunks the widening may put in the prompt; a guardrail
        against a run of long articles, not a target, so it should rarely bite.
    """

    ANTHROPIC_API_KEY: str = ""
    CHAT_MODEL: str = "anthropic/claude-haiku-4-5"
    CHAT_TIMEOUT: int = 60
    CHAT_MAX_TOKENS: int = 2048
    CHAT_SOURCES: int = Field(default=5, ge=1)
    CHAT_CONTEXT_CHUNKS: int = Field(default=15, ge=1)


class IngestConfig(BaseConfig):
    """Ingestion tunables.

    TOPIC_BASE_ACTS: the act each topic is built from; its corpus is everything based on it.
    CRAWL_DELAYS: seconds between requests per host; eur-lex publishes 10 in robots.txt,
        the rest is courtesy.
    MAX_DROP_RATIO: fraction of the previous corpus that may vanish before discovery aborts.
    MIN_SUSPICIOUS_DROPS: dropped documents below this never abort, however small the corpus.
    MAX_CHARS: longest chunk emitted.
    EMBED_PAGE_SIZE: chunks the embed sweep reads per page.
    EMBED_CONCURRENCY: provider calls in flight at once; llm_retry absorbs the rate limits.
    MAX_FAILURE_CHARS: cut for a stored failure message, so one talkative provider cannot
        bloat the run row.
    """

    TOPIC_BASE_ACTS: dict[str, str] = {"fueleu": "32023R1805", "mrv": "32015R0757"}
    CRAWL_DELAYS: dict[str, float] = {"eur-lex.europa.eu": 10.0, "publications.europa.eu": 1.0}
    MAX_DROP_RATIO: float = 0.2
    MIN_SUSPICIOUS_DROPS: int = 3
    MAX_CHARS: int = 2000
    EMBED_PAGE_SIZE: int = 500
    EMBED_CONCURRENCY: int = 4
    MAX_FAILURE_CHARS: int = 500


class RetrievalConfig(BaseConfig):
    """Search tunables.

    SEARCH_CANDIDATES: per-leg candidate pool feeding Reciprocal Rank Fusion.
    SEARCH_DEFAULT_LIMIT: results returned when the caller does not say how many.
    EF_SEARCH_PER_CANDIDATE: how far the HNSW walk looks per candidate wanted; pgvector
        caps the product at 1000.
    RRF_K: fusion damping; a result at some rank scores 1 / (RRF_K + rank).
    RERANK_ENABLED: the cross-encoder's off switch.
    RERANK_MODEL: which cross-encoder rescores the fused results.
    RERANK_TIMEOUT: seconds to wait for the cross-encoder.
    RERANK_POOL: fused results the cross-encoder rescores.
    EXPAND_SECTIONS: section expansion's off switch; a paragraph rarely restates its
        own subject, so the section is the unit that answers.
    MIN_COSINE_SIMILARITY / MIN_RERANKER_RELEVANCE: the refusal gate's bars, cleared by
        the best hit per signal rather than by one hit on both. Set permissively: the
        gate is for junk, and a false refusal costs more than a wasted call.
    """

    SEARCH_CANDIDATES: int = Field(default=50, ge=1)
    SEARCH_DEFAULT_LIMIT: int = Field(default=10, ge=1)
    EF_SEARCH_PER_CANDIDATE: int = Field(default=4, ge=1)
    RRF_K: int = 60

    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "voyage/rerank-2.5"
    RERANK_TIMEOUT: int = 30
    RERANK_POOL: int = 30

    EXPAND_SECTIONS: bool = True

    MIN_COSINE_SIMILARITY: float = Field(default=0.30, ge=0.0)
    MIN_RERANKER_RELEVANCE: float = Field(default=0.45, ge=0.0)


class EvalConfig(BaseConfig):
    """Eval tunables.

    EVAL_DATASET_PATH: the golden dataset, authored cases versioned as JSON in the repo.
    """

    EVAL_DATASET_PATH: Path = BACKEND_ROOT / "app" / "evals" / "golden.json"


class Config(
    AppConfig,
    PostgresConfig,
    EmbeddingConfig,
    ChatConfig,
    IngestConfig,
    RetrievalConfig,
    StorageConfig,
    EvalConfig,
):
    """Combined configuration class for core app functionality."""


config = Config()
