"""Run results: how a case scores itself, and how a run aggregates its cases."""

from app.chat.prompts import REFUSAL_ANSWER
from app.core.config import config
from app.evals.models import RunMetrics, RunResult, RunSettings
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


def test_a_rerank_off_run_advertises_no_reranker_threshold(monkeypatch) -> None:
    """A gate that never applied must not be recorded as though it had."""
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    settings = RunSettings.from_config()

    assert settings.rerank_enabled is False
    assert settings.rerank_model is None
    assert settings.min_reranker_relevance is None


def test_the_settings_record_every_knob_that_moves_a_hit(monkeypatch) -> None:
    """Flip one of these and every number in the file changes; a file that did not record
    it reads as a corpus regression instead of the config change it was."""
    monkeypatch.setattr(config, "RERANK_ENABLED", True)

    settings = RunSettings.from_config()

    assert settings.embed_model == config.EMBED_MODEL
    assert settings.search_candidates == config.SEARCH_CANDIDATES
    assert settings.rrf_k == config.RRF_K
    assert settings.rerank_model == config.RERANK_MODEL


def test_the_table_omits_the_expansion_row_when_expansion_did_not_run(monkeypatch) -> None:
    """With expansion off the sources are the hits, so a second row would credit a layer
    that never widened anything for recall the raw search had already earned."""
    monkeypatch.setattr(config, "EXPAND_SECTIONS", False)

    table = RunResult.from_results((case_result(),), "sha").table()

    assert "after section expansion" not in table
    assert "raw search hits" in table


def test_the_table_shows_the_expansion_row_when_expansion_ran(monkeypatch) -> None:
    monkeypatch.setattr(config, "EXPAND_SECTIONS", True)

    table = RunResult.from_results((case_result(),), "sha").table()

    assert "after section expansion" in table


def test_the_table_names_the_pattern_a_partial_run_scored() -> None:
    table = RunResult.from_results((case_result(),), "sha", case_pattern="fueleu").table()

    assert "cases matching 'fueleu'" in table
