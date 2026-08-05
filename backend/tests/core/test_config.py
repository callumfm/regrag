"""Configuration behaviour the container depends on."""

from pathlib import Path

import pytest

from app.core.config import load_config
from app.core.enums import Environment


def test_config_loads_from_environment_without_an_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The image ships no .env.prod; compose supplies every value as a real env var."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "prod")
    monkeypatch.setenv("DB_HOST", "db")
    monkeypatch.setenv("DB_PASS", "s3cret")
    monkeypatch.setenv("CORS_ORIGINS", '["https://example.com"]')
    monkeypatch.setenv("RAW_DATA_DIR", "/data/raw")

    config = load_config()

    assert config.ENVIRONMENT is Environment.PROD
    assert config.DB_HOST == "db"
    assert config.CORS_ORIGINS == ["https://example.com"]
    assert config.RAW_DATA_DIR == Path("/data/raw")
    assert "@db:5432/regrag" in config.SQLALCHEMY_DATABASE_URI
