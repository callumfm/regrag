"""Retrieval-only scoring: the shared eval measures a run without a model can fill,
plus what the context costs — the columns recall is traded against."""

from collections.abc import Sequence

from app.chat.enums import ChatNode
from app.evals.enums import EvalKind
from app.evals.metrics import (
    compute_expanded_hit_rate,
    compute_expanded_recall,
    compute_gate_refusal_rate,
    compute_mean_node_ms,
    compute_raw_hit_rate,
    compute_raw_recall,
    count_cases_of_kind,
    count_errors,
    count_false_refusals,
    count_refusals_of_a_found_reference,
)
from app.evals.models import EvalResult
from app.evals.tune.models import RetrievalMetrics


def _scored_in_corpus(results: Sequence[EvalResult]) -> list[EvalResult]:
    """The cases the cost columns average over: completed, and authored to have context."""
    return [r for r in results if r.state.error is None and r.case.kind is EvalKind.IN_CORPUS]


def _mean_or_none(values: Sequence[float]) -> float | None:
    """The mean, or None when there is nothing to average — unmeasured, not zero."""
    return sum(values) / len(values) if values else None


def compute_mean_context_chunks(results: Sequence[EvalResult]) -> float | None:
    """Mean context blocks per scored in-corpus case: what the prompt would carry."""
    return _mean_or_none([float(len(r.state.sources)) for r in _scored_in_corpus(results)])


def compute_mean_context_chars(results: Sequence[EvalResult]) -> float | None:
    """Mean context text length per scored in-corpus case: what the recall is bought with."""
    return _mean_or_none(
        [float(sum(len(c.text) for c in r.state.sources)) for r in _scored_in_corpus(results)]
    )


def compute_mean_retrieve_ms(results: Sequence[EvalResult]) -> int:
    """The retrieve node's mean time over the cases that ran it."""
    return compute_mean_node_ms(results).get(ChatNode.RETRIEVE.value, 0)


def compute_retrieval_metrics(results: Sequence[EvalResult]) -> RetrievalMetrics:
    """Every retrieval measure of the run, each computed over the cases it applies to."""
    return RetrievalMetrics(
        cases=len(results),
        in_corpus=count_cases_of_kind(results, EvalKind.IN_CORPUS),
        out_of_corpus=count_cases_of_kind(results, EvalKind.OUT_OF_CORPUS),
        errors=count_errors(results),
        raw_hit_rate=compute_raw_hit_rate(results),
        raw_recall=compute_raw_recall(results),
        expanded_hit_rate=compute_expanded_hit_rate(results),
        expanded_recall=compute_expanded_recall(results),
        gate_refusal_rate=compute_gate_refusal_rate(results),
        false_refusals=count_false_refusals(results),
        refused_a_found_reference=count_refusals_of_a_found_reference(results),
        mean_context_chunks=compute_mean_context_chunks(results),
        mean_context_chars=compute_mean_context_chars(results),
        mean_retrieve_ms=compute_mean_retrieve_ms(results),
    )
