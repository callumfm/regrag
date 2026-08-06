"""Settings classes: inherited config and per-vendor defaults."""

from pydantic_settings import BaseSettings

from app.core.config import BaseConfig, Config, EmbeddingConfig


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


def test_combined_config_carries_every_concern():
    combined = Config()

    assert combined.PROJECT_NAME == "RegRag"
    assert combined.DB_PORT == 5432
    assert combined.EMBED_DIMENSIONS == 1024
