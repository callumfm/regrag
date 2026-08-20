"""What the eval call cache caches, and the request parameters it keys on."""

from pathlib import Path

import litellm
import pytest
from litellm.caching import Cache
from litellm.caching.disk_cache import DiskCache

from app.core.config import config
from app.evals.cache import enable_call_cache


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the cache at a throwaway directory and leave litellm uncached afterwards,
    so an enabled cache never outlives the test that enabled it."""
    directory = tmp_path / "cache" / "evals"
    monkeypatch.setattr(config, "EVAL_CACHE_DIR", directory)
    monkeypatch.setattr(litellm, "cache", None)
    return directory


def enabled_cache() -> Cache:
    """The cache the CLI would install, once it is installed."""
    enable_call_cache()
    assert isinstance(litellm.cache, Cache)
    return litellm.cache


def cached_calls(cache: Cache) -> set[str]:
    return set(cache.supported_call_types or ())


def test_enable_call_cache_caches_to_the_configured_directory(cache_dir: Path) -> None:
    cache = enabled_cache()

    assert cache_dir.is_dir()
    assert isinstance(cache.cache, DiskCache)
    assert cache.cache.disk_cache.directory == str(cache_dir)


def test_enable_call_cache_caches_only_the_paid_retrieval_calls() -> None:
    assert cached_calls(enabled_cache()) == {"aembedding", "arerank"}


def test_enable_call_cache_never_caches_synthesis() -> None:
    """A run replaying its own answers would measure the cache rather than the model."""
    assert not cached_calls(enabled_cache()) & {"completion", "acompletion"}


RERANK_CALL = {
    "model": "voyage/rerank-2.5",
    "query": "which vessels does FuelEU cover?",
    "documents": ["chunk one", "chunk two", "chunk three"],
}
EMBED_CALL = {"model": "voyage/voyage-4-lite", "input": ["which vessels?"], "dimensions": 1024}


def cache_key(cache: Cache, **call: object) -> str:
    """The key litellm itself would file this call under."""
    return cache.get_cache_key(**call)


def test_the_same_call_keys_the_same_both_times() -> None:
    cache = enabled_cache()

    assert cache_key(cache, **RERANK_CALL) == cache_key(cache, **RERANK_CALL)
    assert cache_key(cache, **EMBED_CALL) == cache_key(cache, **EMBED_CALL)


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"documents": ["chunk three", "chunk two", "chunk one"]}, id="reordered"),
        pytest.param({"documents": ["chunk one", "chunk two", "re-ingested"]}, id="re-ingested"),
        pytest.param({"model": "voyage/rerank-2"}, id="model"),
        pytest.param({"query": "a different question"}, id="query"),
    ],
)
def test_a_rerank_key_covers_the_model_the_query_and_the_documents_in_order(
    changed: dict[str, object],
) -> None:
    """The whole design rests on this: anything that would change the provider's answer
    changes the key, so there is nothing to invalidate. A litellm upgrade that dropped
    documents from the key would silently replay stale reranks."""
    cache = enabled_cache()

    assert cache_key(cache, **RERANK_CALL) != cache_key(cache, **{**RERANK_CALL, **changed})


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"input": ["another question"]}, id="query"),
        pytest.param({"model": "voyage/voyage-3"}, id="model"),
    ],
)
def test_an_embed_key_covers_the_model_and_the_query(changed: dict[str, object]) -> None:
    cache = enabled_cache()

    assert cache_key(cache, **EMBED_CALL) != cache_key(cache, **{**EMBED_CALL, **changed})
