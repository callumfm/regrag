"""What the call cache caches, and the request parameters it keys on."""

from pathlib import Path

import litellm
import pytest
from litellm.caching import Cache
from litellm.caching.disk_cache import DiskCache

from app.core.llm.cache import call_cache_enabled, enable_call_cache


@pytest.fixture(autouse=True)
def cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the cache at a throwaway directory and leave litellm as it was found: enabling
    a cache registers callbacks as well as setting one, and neither may outlive its test."""
    directory = tmp_path / "cache"
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "enable_caching_on_provider_specific_optional_params", False)
    for callbacks in ("input_callback", "success_callback", "_async_success_callback"):
        monkeypatch.setattr(litellm, callbacks, list(getattr(litellm, callbacks)))
    return directory


@pytest.fixture
def cache(cache_dir: Path) -> Cache:
    """A cache installed for the two retrieval calls, as the evals CLI installs one."""
    enable_call_cache(cache_dir, calls=["aembedding", "arerank"])
    assert isinstance(litellm.cache, Cache)
    return litellm.cache


def test_enable_call_cache_caches_to_the_given_directory(cache_dir: Path, cache: Cache) -> None:
    assert cache_dir.is_dir()
    assert isinstance(cache.cache, DiskCache)
    assert cache.cache.disk_cache.directory == str(cache_dir)


def test_enable_call_cache_caches_only_the_named_calls(cache: Cache) -> None:
    assert set(cache.supported_call_types or ()) == {"aembedding", "arerank"}


def test_nothing_is_cached_until_a_cache_is_enabled(cache_dir: Path) -> None:
    assert not call_cache_enabled()
    enable_call_cache(cache_dir, calls=["aembedding"])
    assert call_cache_enabled()


RERANK_CALL = {
    "model": "voyage/rerank-2.5",
    "query": "which vessels does FuelEU cover?",
    "documents": ["chunk one", "chunk two", "chunk three"],
    "timeout": 5,
}
EMBED_CALL = {
    "model": "voyage/voyage-4-lite",
    "input": ["which vessels?"],
    "dimensions": 1024,
    "input_type": "query",
    "timeout": 5,
}


def test_the_same_call_keys_the_same_both_times(cache: Cache) -> None:
    assert cache.get_cache_key(**RERANK_CALL) == cache.get_cache_key(**RERANK_CALL)
    assert cache.get_cache_key(**EMBED_CALL) == cache.get_cache_key(**EMBED_CALL)


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
    changed: dict[str, object], cache: Cache
) -> None:
    """The whole design rests on this: anything that would change the provider's answer
    changes the key, so there is nothing to invalidate. A litellm upgrade that dropped
    documents from the key would silently replay stale reranks."""
    assert cache.get_cache_key(**RERANK_CALL) != cache.get_cache_key(**{**RERANK_CALL, **changed})


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"input": ["another question"]}, id="query"),
        pytest.param({"model": "voyage/voyage-3"}, id="model"),
        pytest.param({"input_type": "document"}, id="input-type"),
    ],
)
def test_an_embed_key_covers_the_model_the_query_and_the_input_type(
    changed: dict[str, object], cache: Cache
) -> None:
    """input_type is Voyage's own parameter, and litellm leaves those out of the key unless
    told otherwise: without it a document vector would answer for an identical query."""
    assert cache.get_cache_key(**EMBED_CALL) != cache.get_cache_key(**{**EMBED_CALL, **changed})


def test_a_timeout_change_lands_on_a_different_key(cache: Cache) -> None:
    """The key covers more than the provider's answer depends on: raising a timeout to ride
    out a flake spends the whole cache, embeds and reranks alike."""
    assert cache.get_cache_key(**RERANK_CALL) != cache.get_cache_key(
        **{**RERANK_CALL, "timeout": 9}
    )
    assert cache.get_cache_key(**EMBED_CALL) != cache.get_cache_key(**{**EMBED_CALL, "timeout": 9})
