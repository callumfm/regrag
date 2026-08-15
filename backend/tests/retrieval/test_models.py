import pytest
from pydantic import ValidationError

from app.ingestion.chunk.models import Reference
from app.retrieval.models import ReferenceTarget, RetrievedChunk, SearchRequest


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
    chunk = RetrievedChunk(
        id=1, celex="32015R0757", topic="mrv", citation="Article 3", title=None, text="x"
    )

    assert chunk.references == ()
