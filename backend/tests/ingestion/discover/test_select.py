"""The policy over what the topic query returned: which acts are fetched, and at what version."""

from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.select import (
    is_folded_into_another_act,
    is_in_force,
    latest_own_consolidation,
    select_topic_documents,
)
from app.ingestion.discover.sparql import collect_candidate_acts
from tests.conftest import binding, payload


def documents(topic, p):
    return select_topic_documents(topic, collect_candidate_acts(p))


def act(celex, force=None, cons=()):
    bindings = [binding(celex, force=force)] + [binding(celex, force=force, cons=c) for c in cons]
    return collect_candidate_acts(payload(*bindings))[0]


def test_non_legislation_sectors_filtered():
    p = payload(
        binding("32015R0757", force="1"),
        binding("52024XC07469"),
        binding("52024IP0025"),
        binding("E2021X0415(01)"),
    )
    assert [s.celex for s in documents("mrv", p)] == ["32015R0757"]


def test_not_in_force_filtered():
    p = payload(binding("32016R1927", force="0"), binding("32016R1928", force="1"))
    assert [s.celex for s in documents("mrv", p)] == ["32016R1928"]


def test_folded_amendment_filtered():
    p = payload(
        binding("32023R2776", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
    )
    specs = documents("mrv", p)
    assert [s.celex for s in specs] == ["32015R0757"]
    assert specs[0].candidate_celex == "02015R0757-20240101"


def test_candidate_is_max_own_stem_consolidation():
    p = payload(
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
        binding("32015R0757", force="1", cons="02015R0757-20161216"),
    )
    assert documents("mrv", p)[0].candidate_celex == "02015R0757-20250101"


def test_no_consolidations_gives_none_candidate():
    p = payload(binding("32023R2449", force="1"))
    assert documents("mrv", p)[0].candidate_celex is None


def test_specs_carry_topic_and_source():
    p = payload(binding("32023R1805", force="1"))
    spec = documents("fueleu", p)[0]
    assert spec == DiscoveredDocument(
        topic="fueleu", source="eurlex", celex="32023R1805", candidate_celex=None
    )


def test_is_in_force_only_accepts_the_live_flag():
    assert is_in_force(act("32016R1928", force="1"))
    assert not is_in_force(act("32016R1927", force="0"))
    assert not is_in_force(act("32016R1926"))


def test_an_act_consolidated_only_into_another_act_is_folded():
    assert is_folded_into_another_act(act("32023R2776", force="1", cons=["02015R0757-20240101"]))


def test_an_act_with_its_own_consolidation_is_not_folded():
    assert not is_folded_into_another_act(
        act("32015R0757", force="1", cons=["02015R0757-20240101"])
    )


def test_an_act_with_no_consolidations_is_not_folded():
    assert not is_folded_into_another_act(act("32023R2449", force="1"))


def test_latest_own_consolidation_ignores_another_acts_versions():
    versions = ["02015R0757-20240101", "02015R0757-20250101", "02023R1805-20260101"]
    assert latest_own_consolidation(act("32015R0757", force="1", cons=versions)) == (
        "02015R0757-20250101"
    )
