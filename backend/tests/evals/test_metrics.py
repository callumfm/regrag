"""Eval scoring: what counts as a retrieved reference, a correct citation, a refusal."""

from app.chat.prompts import REFUSAL_ANSWER
from app.evals.metrics import (
    find_cited_markers,
    is_gate_refusal,
    score_citation_precision,
    score_reference_recall,
)
from app.retrieval.models import ReferenceTarget
from tests.conftest import retrieved_chunk

ARTICLE_4 = ReferenceTarget(celex="32023R1805", article="4")
ARTICLE_20 = ReferenceTarget(celex="32023R1805", article="20")
ANNEX_IV = ReferenceTarget(celex="32023R1805", annex="IV")


def test_recall_counts_a_gold_article_the_chunks_cover() -> None:
    assert score_reference_recall((ARTICLE_4,), (retrieved_chunk(),)) == 1.0


def test_recall_is_the_share_of_gold_references_retrieved() -> None:
    assert score_reference_recall((ARTICLE_4, ARTICLE_20), (retrieved_chunk(),)) == 0.5


def test_recall_ignores_the_paragraph_a_chunk_sits_in() -> None:
    """Gold cites Article 4; the chunk is Article 4(1). The ticket scores at article grain."""
    chunk = retrieved_chunk(citation="Article 4(1)", article="4")

    assert score_reference_recall((ARTICLE_4,), (chunk,)) == 1.0


def test_recall_matches_an_article_whatever_its_case() -> None:
    gold = ReferenceTarget(celex="32023R1805", article="4a")
    chunk = retrieved_chunk(article="4A")

    assert score_reference_recall((gold,), (chunk,)) == 1.0


def test_recall_matches_an_annex_target_that_has_no_article() -> None:
    chunk = retrieved_chunk(article=None, annex="IV", citation="Annex IV")

    assert score_reference_recall((ANNEX_IV,), (chunk,)) == 1.0


def test_recall_does_not_match_the_right_division_of_another_act() -> None:
    chunk = retrieved_chunk(celex="32015R0757", article="4")

    assert score_reference_recall((ARTICLE_4,), (chunk,)) == 0.0


def test_recall_is_zero_when_nothing_was_retrieved() -> None:
    assert score_reference_recall((ARTICLE_4,), ()) == 0.0


def test_markers_are_read_in_first_cited_order_without_repeats() -> None:
    assert find_cited_markers("A [2] B [1][2] C [10]") == (2, 1, 10)


def test_no_markers_when_the_answer_cites_nothing() -> None:
    assert find_cited_markers(REFUSAL_ANSWER) == ()


def test_citation_precision_is_one_when_every_cited_block_is_gold() -> None:
    sources = (retrieved_chunk(),)

    assert score_citation_precision("Half of it [1].", sources, (ARTICLE_4,)) == 1.0


def test_citation_precision_is_the_share_of_cited_blocks_that_are_gold() -> None:
    sources = (retrieved_chunk(), retrieved_chunk(id=2, article="99", citation="Article 99"))

    assert score_citation_precision("Both [1][2].", sources, (ARTICLE_4,)) == 0.5


def test_a_marker_past_the_end_of_the_context_counts_against_precision() -> None:
    """The model cited a block it was never given, so the citation is wrong, not skipped."""
    sources = (retrieved_chunk(),)

    assert score_citation_precision("Claims [1] and [7].", sources, (ARTICLE_4,)) == 0.5


def test_citation_precision_is_unmeasured_when_the_answer_cites_nothing() -> None:
    assert score_citation_precision("No markers here.", (retrieved_chunk(),), (ARTICLE_4,)) is None


def test_only_the_fixed_wording_marks_a_gate_refusal() -> None:
    assert is_gate_refusal(REFUSAL_ANSWER)
    assert is_gate_refusal(f"  {REFUSAL_ANSWER}\n")
    assert not is_gate_refusal("Half of the voyage's energy counts [1].")


def test_a_model_worded_decline_is_not_a_gate_refusal() -> None:
    """It cost a model call, so the cheap gate did not fire; scoring it is the judge's job."""
    assert not is_gate_refusal("The context provided does not cover ETS allowances.")
