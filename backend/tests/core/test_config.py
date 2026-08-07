"""Settings classes: inherited config and per-vendor defaults."""

import pytest
from pydantic import ValidationError
from pydantic_settings import BaseSettings

from app.core.config import BaseConfig, Config, EmbeddingConfig, StorageConfig
from app.core.enums import StorageBackend


def r2_config(bucket: str = "regrag-raw") -> StorageConfig:
    """A fully configured R2 storage config, with the bucket overridable to leave it out."""
    return StorageConfig(
        STORAGE_BACKEND=StorageBackend.R2,
        R2_ACCOUNT_ID="acc",
        R2_ACCESS_KEY_ID="key",
        R2_SECRET_ACCESS_KEY="secret",
        R2_BUCKET=bucket,
    )


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
    assert embedding.EMBED_DIMENSIONS == 1024
    assert embedding.EMBED_TIMEOUT == 30


def test_storage_defaults_to_the_local_backend():
    """Dev and tests must never need R2 credentials to run the pipeline."""
    assert StorageConfig().STORAGE_BACKEND is StorageBackend.LOCAL


def test_the_r2_endpoint_is_built_from_the_account_id():
    assert r2_config().R2_ENDPOINT_URL == "https://acc.r2.cloudflarestorage.com"


def test_the_r2_backend_refuses_to_start_half_configured():
    with pytest.raises(ValidationError, match="R2_BUCKET"):
        r2_config(bucket="")


def test_r2_settings_are_not_required_by_the_local_backend():
    assert StorageConfig(STORAGE_BACKEND=StorageBackend.LOCAL, R2_BUCKET="").R2_BUCKET == ""


def test_combined_config_carries_every_concern():
    combined = Config()

    assert combined.PROJECT_NAME == "RegRag"
    assert combined.DB_PORT == 5432
    assert combined.EMBED_DIMENSIONS == 1024
    assert combined.STORAGE_BACKEND is StorageBackend.LOCAL
