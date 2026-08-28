"""Tune test factories and guards shared across the tune test modules."""

import pytest

from app.core.config import EVAL_CONFIG_SECTIONS, get_config_snapshot
from app.evals.tune import cli as tune_cli
from app.evals.tune.models import TuneMetrics, TuneResult, TuneRun


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    """Record whether the command turned the call cache on, without turning it on. Autouse
    so no test here installs a real cache: tune enables one by default, which would put a
    cache under the real data directory and leave it set for whatever runs next."""
    calls: list[bool] = []
    monkeypatch.setattr(tune_cli, "enable_call_cache", lambda: calls.append(True))
    return calls


def metrics(**overrides) -> TuneMetrics:
    defaults = dict(
        cases=20,
        in_corpus=15,
        out_of_corpus=5,
        errors=0,
        raw_hit_rate=1.0,
        raw_recall=0.97,
        expanded_hit_rate=1.0,
        expanded_recall=0.97,
        gate_refusal_rate=1.0,
        false_refusals=0,
        refused_a_found_reference=0,
        mean_context_chunks=14.2,
        mean_context_chars=28100.0,
        mean_retrieve_ms=412,
    )
    return TuneMetrics(**{**defaults, **overrides})


def tune_run(*results: TuneResult) -> TuneRun:
    """A run with a healthy baseline and the given varied results."""
    return TuneRun(
        dataset_sha="a" * 64,
        settings=get_config_snapshot(EVAL_CONFIG_SECTIONS),
        baseline=metrics(),
        results=results,
    )
