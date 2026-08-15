import pytest

from app.retrieval.models import ReferenceTarget


def test_a_bare_act_is_refused_rather_than_dumped() -> None:
    """Following a whole act is a filtered search, not a lookup; it must not return the act."""
    with pytest.raises(ValueError, match="article or annex"):
        ReferenceTarget(celex="32015R0757")


def test_an_article_addresses_a_division() -> None:
    assert ReferenceTarget(celex="32015R0757", article="3").article == "3"


def test_an_unnumbered_annex_addresses_a_division() -> None:
    """RRG-75 writes '' for an act whose one annex carries no number, which is still a target."""
    assert ReferenceTarget(celex="32015R0757", annex="").annex == ""
