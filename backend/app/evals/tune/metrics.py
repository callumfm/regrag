"""Retrieval-only scoring: the shared eval measures a run without a model can fill,
plus what the context costs — the columns recall is traded against."""

from collections.abc import Sequence

from app.chat.enums import ChatNode
from app.evals.metrics import (
    compute_mean_node_ms,
    compute_retrieval_metrics,
    mean_or_none,
    scored_in_corpus,
)
from app.evals.models import EvalResult
from app.evals.tune.models import TuneMetrics


def compute_mean_context_chunks(results: Sequence[EvalResult]) -> float | None:
    """Mean context blocks per scored in-corpus case: what the prompt would carry."""
    return mean_or_none([float(len(r.state.sources)) for r in scored_in_corpus(results)])


def compute_mean_context_chars(results: Sequence[EvalResult]) -> float | None:
    """Mean context text length per scored in-corpus case: what the recall is bought with."""
    return mean_or_none(
        [float(sum(len(c.text) for c in r.state.sources)) for r in scored_in_corpus(results)]
    )


def compute_mean_retrieve_ms(results: Sequence[EvalResult]) -> int | None:
    """The retrieve node's mean time over the cases that ran it; None when none did."""
    return compute_mean_node_ms(results).get(ChatNode.RETRIEVE.value)


def compute_tune_metrics(results: Sequence[EvalResult]) -> TuneMetrics:
    """The shared retrieval block plus the context-cost columns."""
    return TuneMetrics(
        **compute_retrieval_metrics(results).model_dump(),
        mean_context_chunks=compute_mean_context_chunks(results),
        mean_context_chars=compute_mean_context_chars(results),
        mean_retrieve_ms=compute_mean_retrieve_ms(results),
    )
