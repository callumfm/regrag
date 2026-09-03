"""Application configuration loaded from environment variables."""

import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
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
        validate_assignment=True,
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
    DB_PASS: SecretStr = SecretStr("postgres")
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
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS.get_secret_value()}"
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

    VOYAGE_API_KEY: SecretStr = SecretStr("")
    EMBED_MODEL: str = "voyage/voyage-4-lite"
    EMBED_TIMEOUT: int = 30


class ChatConfig(BaseConfig):
    """Chat model configuration for the chat graph.

    CHAT_SOURCES: search hits the answer draws on, each widened to its section.
    CHAT_CONTEXT_CHUNKS: the most chunks the widening may put in the prompt; a guardrail
        against a run of long articles, not a target, so it should rarely bite.
    CHAT_TEMPERATURE: how far the model may stray from its likeliest next token. At 0 it
        takes the likeliest every time, which is what an answer quoting law back wants;
        nothing here is served by sampling variety. Anthropic's frontier models reject the
        parameter outright, so a move off Haiku means dropping this and reaching for
        `output_config.effort` instead — which Haiku in turn does not accept.
    """

    ANTHROPIC_API_KEY: SecretStr = SecretStr("")
    CHAT_MODEL: str = "anthropic/claude-haiku-4-5"
    CHAT_TIMEOUT: int = 60
    CHAT_MAX_TOKENS: int = 2048
    CHAT_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=1.0)
    CHAT_SOURCES: int = Field(default=5, ge=1)
    CHAT_CONTEXT_CHUNKS: int = Field(default=15, ge=1)


class AssessConfig(BaseConfig):
    """The assess ⇄ tools loop, which grows the retrieved context before the answer.

    ASSESS_ENABLED: the loop's off switch; off, the graph answers from retrieval alone.
        On by default: over the 40-case golden dataset (RRG-98) it lifted expanded recall
        0.84 to 0.93 and cited references 0.76 to 0.82, both clear of the cite metric's
        ±0.07 noise, repairing 7 cases and regressing 2, for 2.1x the latency and 2.9x the
        input tokens. The gain sits in the multi-hop cases, which is what the loop is for.
    ASSESS_MODEL: which model reviews the context and asks for the tool calls, separate from
        the one that writes the answer: the two jobs are tuned against different measures.
    ASSESS_MAX_ROUNDS: times assess may ask for tool calls before the answer is written.
        One round reaches everything the context can address, because a block's `cites`
        line carries the address of what it points at; a second round only pays again.
    ASSESS_MAX_CALLS: the most tool calls one round may run; the rest are dropped.
    ASSESS_SEARCH_LIMIT: hits one search tool call brings back.
    ASSESS_FOLLOW_LIMIT: chunks one follow_reference call brings back, so a long division
        cannot spend the whole budget in a single call.
    ASSESS_EXTRA_CHUNKS: the most chunks the loop may add on top of the context retrieve
        produced, whatever its size; at 0 the loop reads the context but never grows it.
    """

    ASSESS_ENABLED: bool = True
    ASSESS_MODEL: str = "anthropic/claude-haiku-4-5"
    ASSESS_MAX_ROUNDS: int = Field(default=1, ge=1)
    ASSESS_MAX_CALLS: int = Field(default=4, ge=1)
    ASSESS_SEARCH_LIMIT: int = Field(default=5, ge=1)
    ASSESS_FOLLOW_LIMIT: int = Field(default=5, ge=1)
    ASSESS_EXTRA_CHUNKS: int = Field(default=10, ge=0)


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
    EXPAND_SECTIONS: widens each hit to its whole section; off by default, as the tune
        showed it doubling context cost for no recall or citation gain.
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

    EXPAND_SECTIONS: bool = False

    MIN_COSINE_SIMILARITY: float = Field(default=0.30, ge=0.0)
    MIN_RERANKER_RELEVANCE: float = Field(default=0.45, ge=0.0)


class EvalConfig(BaseConfig):
    """Eval tunables.

    EVAL_DATASET_PATH: the golden dataset, authored cases versioned as JSON in the repo.
    EVAL_CACHE_DIR: where repeated runs replay their embed and rerank calls from. Here
        rather than in RetrievalConfig, which a run records whole in its settings: a
        cache hit answers exactly as a miss would, so it is not a setting a run compares on.
    """

    EVAL_DATASET_PATH: Path = BACKEND_ROOT / "app" / "evals" / "dataset" / "golden.json"
    EVAL_CACHE_DIR: Path = PROJECT_ROOT / "data" / "cache" / "evals"


class JudgeConfig(BaseConfig):
    """The eval judge: the model that reads an answer and grades it.

    EVAL_JUDGE_MODEL: a model other than the one that wrote the answer, so a model does not
        grade its own habits. Recorded on every run: two runs graded by different judges are
        not comparable. Timeout and max tokens are the chat values until each role has its
        own (RRG-99).
    """

    EVAL_JUDGE_MODEL: str = "anthropic/claude-sonnet-5"


class Config(
    AppConfig,
    PostgresConfig,
    EmbeddingConfig,
    ChatConfig,
    AssessConfig,
    IngestConfig,
    RetrievalConfig,
    StorageConfig,
    EvalConfig,
    JudgeConfig,
):
    """Combined configuration class for core app functionality."""


config = Config()


_CONFIG_SECTIONS = (
    AppConfig,
    PostgresConfig,
    EmbeddingConfig,
    ChatConfig,
    AssessConfig,
    IngestConfig,
    RetrievalConfig,
    StorageConfig,
    EvalConfig,
    JudgeConfig,
)
EVAL_CONFIG_SECTIONS = (
    EmbeddingConfig,
    ChatConfig,
    AssessConfig,
    RetrievalConfig,
    JudgeConfig,
)


def get_config_snapshot(
    sections: tuple[type[BaseConfig], ...] | None = None,
) -> dict[str, Any]:
    """Return non-secret settings from the requested config sections."""
    sections = sections or _CONFIG_SECTIONS

    return {
        name: getattr(config, name)
        for section in sections
        for name, field in sorted(section.model_fields.items())
        if field.annotation is not SecretStr
    }
