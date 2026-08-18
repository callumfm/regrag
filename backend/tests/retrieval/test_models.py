import pytest
from pydantic import ValidationError

from app.ingestion.chunk.models import Reference
from app.retrieval.models import ReferenceTarget, SearchRequest
from tests.conftest import retrieved_chunk, search_result


def test_a_bare_act_is_refused_rather_than_dumped() -> None:
    """Following a whole act is a filtered search, not a lookup; it must not return the act."""
    with pytest.raises(ValueError, match="article or annex"):
        ReferenceTarget(celex="32015R0757")


def test_an_article_addresses_a_division() -> None:
    assert ReferenceTarget(celex="32015R0757", article="3").article == "3"


def test_an_unnumbered_annex_addresses_a_division() -> None:
    """RRG-75 writes '' for an act whose one annex carries no number, which is still a target."""
    assert ReferenceTarget(celex="32015R0757", annex="").annex == ""


def test_a_reference_naming_no_instrument_targets_the_citing_act() -> None:
    reference = Reference(raw="Article 11a", article="11a")

    target = ReferenceTarget.from_reference(reference, citing="32015R0757")

    assert target == ReferenceTarget(celex="32015R0757", article="11a")


def test_a_reference_naming_an_instrument_targets_that_act_and_keeps_its_annex() -> None:
    reference = Reference(
        raw="Annex I to Regulation (EU) 2023/1805", instrument="32023R1805", annex="I"
    )

    target = ReferenceTarget.from_reference(reference, citing="32015R0757")

    assert target == ReferenceTarget(celex="32023R1805", annex="I")


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_is_refused_at_the_request(limit: int) -> None:
    """A non-positive limit would slice results wrongly or hand Postgres a negative LIMIT."""
    with pytest.raises(ValidationError, match="limit"):
        SearchRequest(query="energy", limit=limit)


def test_a_request_may_leave_the_limit_to_config() -> None:
    assert SearchRequest(query="energy").limit is None


def test_a_retrieved_chunk_cites_nothing_unless_told_otherwise() -> None:
    """Every database path fills references, so a hand-built chunk need not spell out none."""
    assert retrieved_chunk().references == ()


def test_a_line_shows_every_signal_then_the_citation() -> None:
    result = search_result(rrf_score=0.0328, cosine_similarity=0.7512, reranker_relevance=0.6)

    assert result.line().startswith("rrf 0.0328  cos  0.7512  rel  0.6000  Article 4(1)")
    assert "32023R1805" in result.line()


def test_a_missing_signal_prints_as_a_dash() -> None:
    result = search_result(cosine_similarity=None, reranker_relevance=None)

    assert result.line().startswith("rrf 0.9000  cos       -  rel       -  Article 4(1)")


def test_a_negative_similarity_keeps_the_columns_in_line() -> None:
    result = search_result(cosine_similarity=-0.0231, reranker_relevance=0.6)

    assert result.line().startswith("rrf 0.9000  cos -0.0231  rel  0.6000  Article 4(1)")
