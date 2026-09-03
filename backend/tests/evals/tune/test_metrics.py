"""Retrieval-only metrics: the shared scorers plus what the context costs."""

from app.evals.tune.metrics import compute_tune_metrics
from tests.evals.conftest import eval_result, refused_result


def test_a_scored_case_fills_the_retrieval_measures() -> None:
    result = eval_result()
    metrics = compute_tune_metrics([result])

    assert metrics.counts.cases == 1
    assert metrics.counts.in_corpus == 1
    assert metrics.retrieval.raw_recall == 1.0
    assert metrics.retrieval.expanded_recall == 1.0
    assert metrics.context.mean_context_chunks == 1.0
    assert metrics.context.mean_context_chars == float(len(result.state.sources[0].text))
    assert metrics.latency.mean_step_ms["retrieve"] == 100


def test_cost_columns_average_scored_in_corpus_cases_only() -> None:
    """A refusal builds no context, so it must not drag the mean toward zero."""
    metrics = compute_tune_metrics([eval_result(), refused_result()])

    assert metrics.counts.out_of_corpus == 1
    assert metrics.gate.refusal_rate == 1.0
    assert metrics.context.mean_context_chunks == 1.0


def test_no_cases_leaves_every_rate_unmeasured() -> None:
    metrics = compute_tune_metrics([])

    assert metrics.counts.cases == 0
    assert metrics.retrieval.raw_recall is None
    assert metrics.context.mean_context_chunks is None
    assert metrics.latency.mean_step_ms == {}
