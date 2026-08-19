"""Run results: how a case scores itself, and how a run aggregates its cases."""

from app.chat.prompts import REFUSAL_ANSWER
from app.evals.models import RunMetrics
from app.retrieval.models import ReferenceTarget
from tests.conftest import retrieved_chunk, search_result
from tests.evals.conftest import REFERENCE, case_result, eval_case, out_of_corpus_case

ARTICLE_20 = ReferenceTarget(celex="32023R1805", article="20")
OTHER_CHUNK = {"id": 2, "article": "99", "citation": "Article 99"}


def test_a_case_scores_the_raw_hits_and_the_expanded_sources_apart() -> None:
    """Expansion found the gold reference the raw hits missed — the layer that did the work."""
    result = case_result(
        hits=(search_result(**OTHER_CHUNK),),
        sources=(retrieved_chunk(**OTHER_CHUNK), retrieved_chunk()),
    )

    assert result.raw_recall == 0.0
    assert result.expanded_recall == 1.0


def test_a_refused_out_of_corpus_case_is_a_correct_refusal() -> None:
    result = case_result(out_of_corpus_case(), hits=(), sources=(), answer=REFUSAL_ANSWER)

    assert result.gate_refused
    assert not result.refused_a_covered_case


def test_a_refusal_over_hits_holding_the_gold_reference_is_the_gate_too_tight() -> None:
    result = case_result(hits=(search_result(),), sources=(), answer=REFUSAL_ANSWER)

    assert result.refused_a_covered_case


def test_a_refusal_over_hits_that_missed_the_gold_reference_is_a_genuine_miss() -> None:
    result = case_result(hits=(search_result(**OTHER_CHUNK),), sources=(), answer=REFUSAL_ANSWER)

    assert result.gate_refused
    assert not result.refused_a_covered_case


def test_hit_rate_and_recall_diverge_on_a_half_served_multi_reference_case() -> None:
    """The whole reason both are reported: one case found 1 of its 2 gold references."""
    served = case_result()
    half = case_result(eval_case(id="two-refs", references=(REFERENCE, ARTICLE_20)))

    metrics = RunMetrics.from_cases((served, half))

    assert metrics.expanded_hit_rate == 1.0
    assert metrics.expanded_recall == 0.75


def test_an_errored_case_is_counted_but_left_out_of_the_scores() -> None:
    """A provider blip must not read as a retrieval regression."""
    metrics = RunMetrics.from_cases(
        (case_result(), case_result(eval_case(id="boom"), hits=(), sources=(), error="timeout"))
    )

    assert metrics.errors == 1
    assert metrics.expanded_hit_rate == 1.0


def test_a_kind_absent_from_the_run_scores_none_rather_than_zero() -> None:
    only_ooc = RunMetrics.from_cases(
        (case_result(out_of_corpus_case(), hits=(), sources=(), answer=REFUSAL_ANSWER),)
    )

    assert only_ooc.expanded_hit_rate is None
    assert only_ooc.gate_refusal_rate == 1.0


def test_refusal_accuracy_and_false_refusals_count_opposite_mistakes() -> None:
    metrics = RunMetrics.from_cases(
        (
            case_result(out_of_corpus_case("ooc-1"), hits=(), sources=(), answer=REFUSAL_ANSWER),
            case_result(out_of_corpus_case("ooc-2"), answer="It is 42 [1]."),
            case_result(eval_case(id="wrongly-refused"), sources=(), answer=REFUSAL_ANSWER),
        )
    )

    assert metrics.gate_refusal_rate == 0.5
    assert metrics.false_refusals == 1


def test_an_errored_case_shows_dashes_rather_than_the_scores_it_never_earned() -> None:
    row = case_result(error="chat call timed out").line()

    assert "raw    -" in row
    assert "1.00" not in row
    assert row.endswith("chat call timed out")


def test_an_out_of_corpus_row_has_no_recall_to_show() -> None:
    row = case_result(out_of_corpus_case(), hits=(), sources=(), answer=REFUSAL_ANSWER).line()

    assert "raw    -  exp    -" in row
    assert "gate-refused" in row
