"""Settings classes: inherited config and per-vendor defaults."""

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings

from app.core.config import (
    BACKEND_ROOT,
    EVAL_CONFIG_SECTIONS,
    AssessConfig,
    BaseConfig,
    ChatConfig,
    Config,
    EmbeddingConfig,
    Environment,
    IngestConfig,
    RetrievalConfig,
    StorageBackend,
    StorageConfig,
    config,
    get_config_snapshot,
    get_env_file,
    load_environment,
)
from tests.conftest import r2_config


def test_storage_defaults_to_the_local_backend():
    assert StorageConfig().STORAGE_BACKEND is StorageBackend.LOCAL


def test_the_r2_endpoint_is_built_from_the_account_id(monkeypatch):
    assert r2_config(monkeypatch).R2_ENDPOINT_URL == "https://acc.r2.cloudflarestorage.com"


def test_every_r2_setting_is_required(monkeypatch):
    """A half-configured bucket must fail when the store is built, not mid-run."""
    with pytest.raises(ValidationError, match="R2_BUCKET"):
        r2_config(monkeypatch, R2_BUCKET="")


def test_the_combined_config_does_not_carry_the_r2_settings():
    """Dev and tests must never need R2 credentials to load configuration."""
    combined = Config()

    assert not hasattr(combined, "R2_BUCKET")
    assert combined.STORAGE_BACKEND is StorageBackend.LOCAL


def test_subclass_inherits_settings_without_restating_them():
    class Sample(BaseConfig):
        SOME_VALUE: str = "default"

    sample = Sample()

    assert sample.model_config["extra"] == "ignore"
    assert sample.model_config["env_ignore_empty"] is True
    assert sample.model_config["env_file"] == BaseConfig.model_config["env_file"]


def test_base_config_is_a_settings_class():
    assert issubclass(BaseConfig, BaseSettings)


def test_embedding_defaults_match_voyage_4_lite(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedding = EmbeddingConfig()

    assert embedding.EMBED_MODEL == "voyage/voyage-4-lite"
    assert embedding.EMBED_TIMEOUT == 30


def test_the_embedding_width_is_not_settable_from_the_environment(monkeypatch):
    """It has to match the deployed vector column, so no env var may move it."""
    monkeypatch.setenv("EMBED_DIMENSIONS", "1536")

    assert not hasattr(EmbeddingConfig(), "EMBED_DIMENSIONS")


def test_combined_config_carries_every_concern():
    combined = Config()

    assert combined.PROJECT_NAME == "RegRag"
    assert combined.DB_PORT == 5432
    assert combined.EMBED_MODEL == "voyage/voyage-4-lite"


def test_an_unrecognised_environment_names_the_accepted_values(monkeypatch):
    """A host exporting its own ENVIRONMENT must not crash-loop on a bare enum traceback."""
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(ValueError, match="ENVIRONMENT='production'.*dev, test, prod"):
        load_environment()


def test_environment_defaults_to_dev_when_unset(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    assert load_environment() is Environment.DEV


@pytest.mark.parametrize(
    ("env", "filename"),
    [
        (Environment.DEV, ".env.dev"),
        (Environment.TEST, ".env.example"),
        (Environment.PROD, ".env.prod"),
    ],
)
def test_the_env_file_is_absolute_so_the_working_directory_cannot_change_it(env, filename):
    env_file = get_env_file(env)

    assert env_file == BACKEND_ROOT / filename
    assert env_file.is_absolute()


def test_the_suite_runs_against_a_database_of_its_own():
    """Tests commit destructive deletes, so reaching the dev database would cost a corpus."""
    assert Config().DB_NAME == "regrag_test"


def test_ingest_defaults_match_the_shipped_tunables():
    ingest = IngestConfig()

    assert ingest.TOPIC_BASE_ACTS == {"fueleu": "32023R1805", "mrv": "32015R0757"}
    assert ingest.CRAWL_DELAYS == {"eur-lex.europa.eu": 10.0, "publications.europa.eu": 1.0}
    assert ingest.MAX_DROP_RATIO == 0.2
    assert ingest.MIN_SUSPICIOUS_DROPS == 3
    assert ingest.MAX_CHARS == 2000
    assert ingest.EMBED_PAGE_SIZE == 500
    assert ingest.EMBED_CONCURRENCY == 4
    assert ingest.MAX_FAILURE_CHARS == 500


def test_combined_config_carries_the_ingest_tunables():
    assert Config().EMBED_CONCURRENCY == 4
    assert sorted(Config().TOPIC_BASE_ACTS) == ["fueleu", "mrv"]


def test_retrieval_defaults_match_the_shipped_tunables():
    retrieval = RetrievalConfig()

    assert retrieval.SEARCH_CANDIDATES == 50
    assert retrieval.SEARCH_DEFAULT_LIMIT == 10
    assert retrieval.EF_SEARCH_PER_CANDIDATE == 4
    assert retrieval.RRF_K == 60
    assert retrieval.RERANK_ENABLED is True
    assert retrieval.RERANK_MODEL == "voyage/rerank-2.5"
    assert retrieval.RERANK_TIMEOUT == 30
    assert retrieval.RERANK_POOL == 30
    assert retrieval.EXPAND_SECTIONS is False


@pytest.mark.parametrize(
    "name", ["SEARCH_CANDIDATES", "SEARCH_DEFAULT_LIMIT", "EF_SEARCH_PER_CANDIDATE"]
)
def test_a_search_knob_of_zero_is_refused_at_startup(name, monkeypatch):
    """Zero reaches Postgres as an ef_search or LIMIT it rejects, so it fails before any query."""
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValidationError, match=name):
        RetrievalConfig()


def test_combined_config_carries_the_retrieval_tunables():
    assert Config().RERANK_POOL == 30
    assert Config().RERANK_ENABLED is True


def test_chat_defaults():
    chat = ChatConfig()
    assert chat.CHAT_MODEL == "anthropic/claude-haiku-4-5"
    assert chat.CHAT_TIMEOUT == 60
    assert chat.CHAT_MAX_TOKENS == 2048
    assert chat.CHAT_SOURCES == 5
    assert chat.CHAT_CONTEXT_CHUNKS == 15
    assert chat.ANTHROPIC_API_KEY.get_secret_value() == ""


def test_config_includes_chat_settings():
    assert "CHAT_MODEL" in Config.model_fields


def test_assess_defaults():
    assess = AssessConfig()
    assert assess.ASSESS_ENABLED is False
    assert assess.ASSESS_MODEL == "anthropic/claude-haiku-4-5"
    assert assess.ASSESS_MAX_ROUNDS == 2
    assert assess.ASSESS_MAX_CALLS == 4
    assert assess.ASSESS_SEARCH_LIMIT == 5
    assert assess.ASSESS_EXTRA_CHUNKS == 10


def test_the_loop_needs_at_least_one_round_when_it_is_on():
    """ASSESS_ENABLED is the off switch, so a round budget of zero is a misconfiguration
    rather than a second way to say off."""
    with pytest.raises(ValidationError, match="ASSESS_MAX_ROUNDS"):
        AssessConfig(ASSESS_MAX_ROUNDS=0)


# What a run records about the settings it read


def test_a_snapshot_records_the_requested_sections_whole(monkeypatch):
    """Taken from the config sections whole, so a knob added later is recorded without
    anyone editing a list."""
    monkeypatch.setattr(config, "CHAT_CONTEXT_CHUNKS", 7)

    settings = get_config_snapshot(EVAL_CONFIG_SECTIONS)

    expected = {name for section in EVAL_CONFIG_SECTIONS for name in section.model_fields}
    assert set(settings) == expected - {"VOYAGE_API_KEY", "ANTHROPIC_API_KEY"}
    assert settings["CHAT_CONTEXT_CHUNKS"] == 7
    assert settings["RERANK_POOL"] == config.RERANK_POOL


def test_a_snapshot_leaves_out_the_secrets(monkeypatch):
    """The snapshot is printed and pasted around; a key is not a knob a run reproduces."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", SecretStr("sk-never-recorded"))

    snapshot = get_config_snapshot()

    assert "sk-never-recorded" not in str(snapshot.values())
    assert not any("API_KEY" in name for name in snapshot)
    assert "DB_PASS" not in snapshot


def test_assignment_is_validated_and_coerced_by_the_field():
    combined = Config()
    setattr(combined, "CHAT_SOURCES", "3")  # noqa: B010 — a deliberately mistyped write

    assert combined.CHAT_SOURCES == 3
    with pytest.raises(ValidationError):
        combined.CHAT_SOURCES = 0
