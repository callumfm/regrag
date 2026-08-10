"""CELLAR discovery: SPARQL response parsing, filters, seed guard."""

from pathlib import Path

import httpx
import pytest

from app.ingestion.constants import SEEDS
from app.ingestion.exceptions import MalformedDiscoveryError
from app.ingestion.fetch.discover import (
    collect_candidate_acts,
    discover_topic,
    find_dropped_celexes,
    is_folded_into_another_act,
    is_in_force,
    latest_own_consolidation,
    select_topic_documents,
    topic_query,
    version_candidates,
)
from app.ingestion.fetch.download import download_fetchable_version
from app.ingestion.fetch.models import DiscoveredDocument

FIXTURES = Path(__file__).parent / "fixtures"

EXPECTED_RESOLVED = {
    "fueleu:32023R1805": "32023R1805",
    "fueleu:32024R2027": "32024R2027",
    "fueleu:32024R2031": "32024R2031",
    "fueleu:32025R0192": "32025R0192",
    "fueleu:32025R1127": "32025R1127",
    "fueleu:32026R0394": "32026R0394",
    "mrv:32015R0757": "02015R0757-20250101",
    "mrv:32016R1928": "32016R1928",
    "mrv:32023R2449": "32023R2449",
    "mrv:32023R2849": "32023R2849",
    "mrv:32023R2917": "32023R2917",
}

MISSING_HTML = {"02023R1805-20230922", "02023R2917-20231229", "02024R2027-20240729"}


def binding(celex, force=None, cons=None):
    b = {"c": {"value": celex}}
    if force is not None:
        b["force"] = {"value": force}
    if cons is not None:
        b["cons"] = {"value": cons}
    return b


def payload(*bindings):
    return {"results": {"bindings": list(bindings)}}


def documents(topic, p):
    return select_topic_documents(topic, collect_candidate_acts(p))


def act(celex, force=None, cons=()):
    bindings = [binding(celex, force=force)] + [binding(celex, force=force, cons=c) for c in cons]
    return collect_candidate_acts(payload(*bindings))[0]


def spec(celex, topic="mrv"):
    return DiscoveredDocument(topic=topic, source="eurlex", celex=celex, candidate_celex=None)


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


def test_version_candidates_try_the_consolidation_before_the_original_act():
    consolidated = DiscoveredDocument(
        topic="mrv", source="eurlex", celex="32015R0757", candidate_celex="02015R0757-20250101"
    )
    assert version_candidates(consolidated) == ["02015R0757-20250101", "32015R0757"]


def test_version_candidates_without_a_consolidation_is_the_act_alone():
    assert version_candidates(spec("32023R2449")) == ["32023R2449"]


def test_specs_carry_topic_and_source():
    p = payload(binding("32023R1805", force="1"))
    spec = documents("fueleu", p)[0]
    assert spec == DiscoveredDocument(
        topic="fueleu", source="eurlex", celex="32023R1805", candidate_celex=None
    )


def test_topic_query_embeds_seed():
    q = topic_query("32023R1805")
    assert "resource/celex/32023R1805" in q
    assert "resource_legal_based_on_resource_legal" in q


def test_discover_raises_when_seed_missing_from_results():
    def handler(request):
        return httpx.Response(200, json=payload(binding("32023R2449", force="1")))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MalformedDiscoveryError, match="32015R0757"):
            discover_topic(client, "mrv", "32015R0757")


def test_discover_returns_parsed_specs():
    def handler(request):
        assert request.url.params["format"] == "application/sparql-results+json"
        return httpx.Response(
            200,
            json=payload(
                binding("32015R0757", force="1", cons="02015R0757-20250101"),
                binding("32023R2449", force="1"),
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        specs = discover_topic(client, "mrv", "32015R0757")
    assert [s.celex for s in specs] == ["32015R0757", "32023R2449"]


DOC_HTML = (FIXTURES / "doc.html").read_text()
MISSING_HTML_PAGE = (FIXTURES / "missing.html").read_text()
SPARQL_FIXTURES = {topic: (FIXTURES / f"sparql-{topic}.json").read_text() for topic in SEEDS}


def corpus_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "publications.europa.eu":
        query = request.url.params["query"]
        for topic, seed in SEEDS.items():
            if seed in query:
                return httpx.Response(200, text=SPARQL_FIXTURES[topic])
        raise AssertionError(f"no seed in query: {query[:120]}")
    celex = request.url.params["uri"].removeprefix("CELEX:")
    if celex in MISSING_HTML:
        return httpx.Response(404, text=MISSING_HTML_PAGE)
    return httpx.Response(200, text=DOC_HTML)


@pytest.mark.parametrize("topic", sorted(SEEDS))
def test_topic_corpus_discovers_and_resolves(topic):
    with httpx.Client(transport=httpx.MockTransport(corpus_handler)) as client:
        specs = discover_topic(client, topic, SEEDS[topic])
        resolved = {
            f"{topic}:{s.celex}": download_fetchable_version(client, s)[0].resolved_celex
            for s in specs
        }
    expected = {k: v for k, v in EXPECTED_RESOLVED.items() if k.startswith(f"{topic}:")}
    assert resolved == expected


def test_collect_candidate_acts_folds_every_binding_for_one_celex():
    p = payload(
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
    )
    acts = collect_candidate_acts(p)
    assert len(acts) == 1
    assert acts[0].consolidations == frozenset({"02015R0757-20240101", "02015R0757-20250101"})


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


def test_find_dropped_celexes_returns_baseline_celexes_absent_from_discovery():
    found = [spec("32015R0757"), spec("32016R1928")]
    baseline = ["32015R0757", "32016R1928", "32014R0666"]
    assert find_dropped_celexes(found, baseline) == ["32014R0666"]


def test_find_dropped_celexes_is_empty_when_all_are_discovered():
    assert find_dropped_celexes([spec("32015R0757")], ["32015R0757"]) == []
