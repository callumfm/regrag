"""Retrieval-only scoring: the shared eval measures a run without a model can fill,
plus what the context costs — the columns recall is traded against."""

from collections.abc import Sequence

from app.evals.metrics import (
    compute_case_counts,
    compute_gate_metrics,
    compute_latency_metrics,
    compute_retrieval_metrics,
    mean_or_none,
    scored_in_corpus,
)
from app.evals.models import EvalResult
from app.evals.tune.models import ContextMetrics, TuneMetrics


def compute_mean_context_chunks(results: Sequence[EvalResult]) -> float | None:
    """Mean context blocks per scored in-corpus case: what the prompt would carry."""
    return mean_or_none([float(len(r.state.sources)) for r in scored_in_corpus(results)])


def compute_mean_context_chars(results: Sequence[EvalResult]) -> float | None:
    """Mean context text length per scored in-corpus case: what the recall is bought with."""
    return mean_or_none(
        [float(sum(len(c.text) for c in r.state.sources)) for r in scored_in_corpus(results)]
    )


def compute_context_metrics(results: Sequence[EvalResult]) -> ContextMetrics:
    return ContextMetrics(
        mean_context_chunks=compute_mean_context_chunks(results),
        mean_context_chars=compute_mean_context_chars(results),
    )


def compute_tune_metrics(results: Sequence[EvalResult]) -> TuneMetrics:
    """The shared blocks plus the context-cost block."""
    return TuneMetrics(
        counts=compute_case_counts(results),
        retrieval=compute_retrieval_metrics(results),
        gate=compute_gate_metrics(results),
        latency=compute_latency_metrics(results),
        context=compute_context_metrics(results),
    )
