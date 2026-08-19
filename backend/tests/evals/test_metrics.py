"""Eval scoring: what counts as a retrieved reference, a correct citation, a refusal."""

from app.chat.prompts import REFUSAL_ANSWER
from app.evals.metrics import (
    find_cited_markers,
    score_citation_validity,
    score_reference_citation_rate,
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


def test_an_unnumbered_annex_does_not_match_a_chunk_outside_every_annex() -> None:
    """Locator documents "" as the act's one unnumbered annex — a different fact from None,
    which no truthiness test may fold together."""
    unnumbered = ReferenceTarget(celex="32023R1805", annex="")
    outside = retrieved_chunk(article=None, annex=None, citation="Preamble")

    assert score_reference_recall((unnumbered,), (outside,)) == 0.0
    assert score_reference_recall((unnumbered,), (retrieved_chunk(article=None, annex=""),)) == 1.0


def test_recall_compares_an_annex_exactly_as_follow_does() -> None:
    """`follow._targeted` case-folds the article but matches the annex exactly, so scoring
    must too — otherwise `check` calls a reference stale that `run` calls recalled."""
    chunk = retrieved_chunk(article=None, annex="iv", citation="Annex iv")

    assert score_reference_recall((ANNEX_IV,), (chunk,)) == 0.0


def test_recall_does_not_match_the_right_division_of_another_act() -> None:
    chunk = retrieved_chunk(celex="32015R0757", article="4")

    assert score_reference_recall((ARTICLE_4,), (chunk,)) == 0.0


def test_recall_is_zero_when_nothing_was_retrieved() -> None:
    assert score_reference_recall((ARTICLE_4,), ()) == 0.0


def test_markers_are_read_in_first_cited_order_without_repeats() -> None:
    assert find_cited_markers("A [2] B [1][2] C [10]") == (2, 1, 10)


def test_no_markers_when_the_answer_cites_nothing() -> None:
    assert find_cited_markers(REFUSAL_ANSWER) == ()


def test_every_authored_reference_cited_scores_one() -> None:
    sources = (retrieved_chunk(),)

    assert score_reference_citation_rate("Half of it [1].", sources, (ARTICLE_4,)) == 1.0


def test_citing_a_further_relevant_article_is_not_penalised() -> None:
    """The authored set names the references an answer must lean on, not every chunk that
    may support it, so an extra citation must not read as a wrong one."""
    sources = (retrieved_chunk(), retrieved_chunk(id=2, article="99", citation="Article 99"))

    assert score_reference_citation_rate("Both [1][2].", sources, (ARTICLE_4,)) == 1.0


def test_the_reference_citation_rate_is_the_share_of_authored_references_cited() -> None:
    sources = (retrieved_chunk(), retrieved_chunk(id=2, article="20", citation="Article 20"))

    assert score_reference_citation_rate("One [1].", sources, (ARTICLE_4, ARTICLE_20)) == 0.5


def test_an_answer_citing_none_of_the_authored_references_scores_zero() -> None:
    sources = (retrieved_chunk(id=2, article="99", citation="Article 99"),)

    assert score_reference_citation_rate("Elsewhere [1].", sources, (ARTICLE_4,)) == 0.0


def test_the_reference_citation_rate_is_unmeasured_when_a_case_names_no_reference() -> None:
    assert score_reference_citation_rate("Anything [1].", (retrieved_chunk(),), ()) is None


def test_validity_is_one_when_every_marker_addresses_a_given_block() -> None:
    assert score_citation_validity("Half of it [1].", (retrieved_chunk(),)) == 1.0


def test_a_marker_past_the_end_of_the_context_counts_against_validity() -> None:
    """The model cited a block it was never given, so the citation is wrong, not skipped."""
    assert score_citation_validity("Claims [1] and [7].", (retrieved_chunk(),)) == 0.5


def test_citation_validity_is_unmeasured_when_the_answer_cites_nothing() -> None:
    assert score_citation_validity("No markers here.", (retrieved_chunk(),)) is None
