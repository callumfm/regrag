"""Settings classes: inherited config and per-vendor defaults."""

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from app.core.config import (
    BACKEND_ROOT,
    BaseConfig,
    Config,
    EmbeddingConfig,
    Environment,
    RerankConfig,
    RetrievalConfig,
    StorageBackend,
    StorageConfig,
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


def test_rerank_defaults_to_on_with_voyage_2_5():
    rerank = RerankConfig()

    assert rerank.RERANK_ENABLED is True
    assert rerank.RERANK_MODEL == "voyage/rerank-2.5"
    assert rerank.RERANK_TIMEOUT == 30


def test_combined_config_carries_the_rerank_settings():
    assert Config().RERANK_ENABLED is True


def test_retrieval_defaults_match_the_shipped_tunables():
    retrieval = RetrievalConfig()

    assert retrieval.SEARCH_CANDIDATES == 50
    assert retrieval.SEARCH_DEFAULT_LIMIT == 10
    assert retrieval.RRF_K == 60
    assert retrieval.RERANK_POOL == 30


def test_combined_config_carries_the_retrieval_tunables():
    assert Config().RERANK_POOL == 30
