"""Tune test factories shared across the tune test modules."""

from app.evals.tune.models import RetrievalMetrics


def metrics(**overrides) -> RetrievalMetrics:
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
    return RetrievalMetrics(**{**defaults, **overrides})
