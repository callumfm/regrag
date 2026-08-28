"""The curated params: every name and value still fits the live config."""

from app.core.config import Config
from app.evals.tune.params import TUNABLE_PARAMS


def test_every_curated_param_and_value_still_fits_the_config() -> None:
    """The drift gate: a renamed or retightened setting fails here, not mid-sweep."""
    for param in TUNABLE_PARAMS:
        param.validate_config()

    assert all(param.name in Config.model_fields for param in TUNABLE_PARAMS)


def test_the_chunk_cap_sweeps_only_with_expansion_on() -> None:
    """CHAT_CONTEXT_CHUNKS is read only inside the expansion branch; sweeping it against
    an expansion-off baseline would re-measure the baseline under another name."""
    chunk_cap = next(param for param in TUNABLE_PARAMS if param.name == "CHAT_CONTEXT_CHUNKS")

    assert chunk_cap.requires == {"EXPAND_SECTIONS": True}


def test_the_reranker_bar_sweeps_only_with_rerank_on() -> None:
    """meets_thresholds applies the bar only to hits carrying a reranker score, so with
    rerank off the sweep would silently re-measure the baseline."""
    reranker_bar = next(param for param in TUNABLE_PARAMS if param.name == "MIN_RERANKER_RELEVANCE")

    assert reranker_bar.requires == {"RERANK_ENABLED": True}
