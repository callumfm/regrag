"""Run results: how a case scores itself, and how a run aggregates its cases."""

from app.chat.enums import ChatNode
from app.core.config import config
from app.evals.results import RunMetrics, RunResult, RunSettings
from app.retrieval.models import ReferenceTarget
from tests.conftest import retrieved_chunk, search_result
from tests.evals.conftest import REFERENCE, case_result, eval_case, out_of_corpus_case

ARTICLE_20 = ReferenceTarget(celex="32023R1805", article="20")
OTHER_CHUNK = {"id": 2, "article": "99", "citation": "Article 99"}
REFUSED = (ChatNode.RETRIEVE, ChatNode.REFUSE)


def test_a_case_scores_the_raw_hits_and_the_expanded_sources_apart() -> None:
    """Expansion found the authored reference the raw hits missed."""
    result = case_result(
        hits=(search_result(**OTHER_CHUNK),),
        sources=(retrieved_chunk(**OTHER_CHUNK), retrieved_chunk()),
    )

    assert result.raw_recall == 0.0
    assert result.expanded_recall == 1.0


def test_a_refusal_is_read_from_the_node_that_ran_not_the_wording() -> None:
    """The graph says which node answered; matching prompt text would break on a reword."""
    result = case_result(out_of_corpus_case(), nodes=REFUSED, hits=(), sources=(), answer="")

    assert result.gate_refused
    assert not result.refused_a_covered_case


def test_an_answer_worded_like_a_refusal_is_not_a_gate_refusal() -> None:
    """Synthesize ran, so the model declined in its own words — the judge's to score."""
    result = case_result(answer="The context provided does not cover ETS allowances.")

    assert not result.gate_refused


def test_a_refusal_over_hits_holding_the_authored_reference_is_the_gate_too_tight() -> None:
    result = case_result(nodes=REFUSED, hits=(search_result(),), sources=(), answer="")

    assert result.refused_a_covered_case


def test_a_refusal_over_hits_that_missed_the_authored_reference_is_a_genuine_miss() -> None:
    result = case_result(nodes=REFUSED, hits=(search_result(**OTHER_CHUNK),), sources=())

    assert result.gate_refused
    assert not result.refused_a_covered_case


def test_hit_rate_and_recall_diverge_on_a_half_served_multi_reference_case() -> None:
    """The whole reason both are reported: one case found 1 of its 2 authored references."""
    half = case_result(eval_case(id="two-refs", references=(REFERENCE, ARTICLE_20)))

    metrics = RunMetrics.from_cases((case_result(), half))

    assert metrics.retrieval.expanded_hit_rate == 1.0
    assert metrics.retrieval.expanded_recall == 0.75


def test_an_errored_case_is_counted_but_left_out_of_the_scores() -> None:
    """A provider blip must not read as a retrieval regression."""
    metrics = RunMetrics.from_cases(
        (case_result(), case_result(eval_case(id="boom"), hits=(), sources=(), error="timeout"))
    )

    assert metrics.errors == 1
    assert metrics.retrieval.expanded_hit_rate == 1.0


def test_a_kind_absent_from_the_run_scores_none_rather_than_zero() -> None:
    only_ooc = RunMetrics.from_cases(
        (case_result(out_of_corpus_case(), nodes=REFUSED, hits=(), sources=(), answer=""),)
    )

    assert only_ooc.retrieval.expanded_hit_rate is None
    assert only_ooc.refusals.gate_rate == 1.0


def test_gate_rate_and_false_refusals_count_opposite_mistakes() -> None:
    metrics = RunMetrics.from_cases(
        (
            case_result(out_of_corpus_case("ooc-1"), nodes=REFUSED, hits=(), sources=(), answer=""),
            case_result(out_of_corpus_case("ooc-2"), answer="It is 42 [1]."),
            case_result(eval_case(id="wrongly-refused"), nodes=REFUSED, sources=(), answer=""),
        )
    )

    assert metrics.refusals.gate_rate == 0.5
    assert metrics.refusals.false_refusals == 1


def test_a_case_citing_none_of_its_references_is_averaged_in_not_dropped() -> None:
    """A rate of 0.0 is a measurement; truthiness would silently discard it."""
    missed = case_result(eval_case(id="missed"), answer="Nothing relevant [1].", sources=())

    metrics = RunMetrics.from_cases((case_result(), missed))

    assert metrics.citations.cited_references == 0.5


def test_the_usage_and_latency_groups_sum_and_average_the_scored_cases() -> None:
    metrics = RunMetrics.from_cases((case_result(), case_result(eval_case(id="second"))))

    assert metrics.usage.input_tokens == 2400
    assert metrics.latency.mean_total_ms == 1000


def test_a_rerank_off_run_advertises_no_reranker_threshold(monkeypatch) -> None:
    """A gate that never applied must not be recorded as though it had."""
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    settings = RunSettings.from_config()

    assert settings.rerank_enabled is False
    assert settings.rerank_model is None
    assert settings.min_reranker_relevance is None


def test_the_settings_record_every_knob_that_moves_a_hit(monkeypatch) -> None:
    monkeypatch.setattr(config, "RERANK_ENABLED", True)

    settings = RunSettings.from_config()

    assert settings.embed_model == config.EMBED_MODEL
    assert settings.search_candidates == config.SEARCH_CANDIDATES
    assert settings.rrf_k == config.RRF_K
    assert settings.rerank_model == config.RERANK_MODEL


def test_the_summary_carries_the_settings_and_scores_but_not_every_case() -> None:
    """Cases hold whole chunks; dumping them would bury the numbers the run is for."""
    summary = RunResult.from_results((case_result(),), "sha123").summary()

    assert '"chat_model"' in summary
    assert '"retrieval"' in summary
    assert '"sha123"' in summary
    assert "The limit applies [1]." not in summary


def test_the_summary_names_every_case_the_graph_raised_on() -> None:
    run = RunResult.from_results(
        (case_result(), case_result(eval_case(id="boom"), error="RuntimeError: pool exhausted")),
        "sha123",
    )

    summary = run.summary()

    assert "errored:" in summary
    assert "boom  RuntimeError: pool exhausted" in summary


def test_a_clean_run_says_nothing_about_errors() -> None:
    assert "errored:" not in RunResult.from_results((case_result(),), "sha123").summary()
