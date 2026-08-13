"""The policy over what the topic query returned: which acts are fetched, and at what version."""

from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.select import select_topic_documents
from app.ingestion.discover.sparql import extract_acts
from tests.conftest import binding


def documents(topic, *rows):
    return select_topic_documents(topic, extract_acts(list(rows)))


def test_non_legislation_sectors_filtered():
    selected = documents(
        "mrv",
        binding("32015R0757", force="1"),
        binding("52024XC07469"),
        binding("52024IP0025"),
        binding("E2021X0415(01)"),
    )
    assert [s.celex for s in selected] == ["32015R0757"]


def test_only_acts_flagged_in_force_are_fetched():
    """Repealed ('0') and unstated (the flag never bound) are both left out."""
    selected = documents(
        "mrv",
        binding("32016R1927", force="0"),
        binding("32016R1926"),
        binding("32016R1928", force="1"),
    )
    assert [s.celex for s in selected] == ["32016R1928"]


def test_folded_amendment_filtered():
    """32023R2776 consolidates only into another act, so its text already lives in 32015R0757."""
    selected = documents(
        "mrv",
        binding("32023R2776", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
    )
    assert [s.celex for s in selected] == ["32015R0757"]
    assert selected[0].candidate_celex == "02015R0757-20240101"


def test_candidate_is_max_own_stem_consolidation():
    selected = documents(
        "mrv",
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
        binding("32015R0757", force="1", cons="02015R0757-20161216"),
    )
    assert selected[0].candidate_celex == "02015R0757-20250101"


def test_candidate_ignores_another_acts_consolidations_even_when_they_sort_higher():
    selected = documents(
        "mrv",
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
        binding("32015R0757", force="1", cons="02023R1805-20260101"),
    )
    assert selected[0].candidate_celex == "02015R0757-20250101"


def test_no_consolidations_gives_none_candidate():
    """An act nothing has consolidated is not folded into anything, so it is still fetched."""
    selected = documents("mrv", binding("32023R2449", force="1"))
    assert [s.celex for s in selected] == ["32023R2449"]
    assert selected[0].candidate_celex is None


def test_specs_carry_topic_and_source():
    spec = documents("fueleu", binding("32023R1805", force="1"))[0]
    assert spec == DiscoveredDocument(
        topic="fueleu", source="eurlex", celex="32023R1805", candidate_celex=None
    )
