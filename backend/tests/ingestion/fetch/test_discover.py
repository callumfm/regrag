"""CELLAR discovery: SPARQL response parsing, filters, seed guard."""

from pathlib import Path

import httpx
import pytest

from app.ingestion.exceptions import DiscoveryError
from app.ingestion.fetch.discover import (
    SEEDS,
    DocumentSpec,
    discover,
    parse_topic_response,
    topic_query,
)
from app.ingestion.fetch.resolve import resolve

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


def test_non_legislation_sectors_filtered():
    p = payload(
        binding("32015R0757", force="1"),
        binding("52024XC07469"),
        binding("52024IP0025"),
        binding("E2021X0415(01)"),
    )
    assert [s.ref for s in parse_topic_response("mrv", p)] == ["32015R0757"]


def test_not_in_force_filtered():
    p = payload(binding("32016R1927", force="0"), binding("32016R1928", force="1"))
    assert [s.ref for s in parse_topic_response("mrv", p)] == ["32016R1928"]


def test_folded_amendment_filtered():
    p = payload(
        binding("32023R2776", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
    )
    specs = parse_topic_response("mrv", p)
    assert [s.ref for s in specs] == ["32015R0757"]
    assert specs[0].candidate_ref == "02015R0757-20240101"


def test_candidate_is_max_own_stem_consolidation():
    p = payload(
        binding("32015R0757", force="1", cons="02015R0757-20240101"),
        binding("32015R0757", force="1", cons="02015R0757-20250101"),
        binding("32015R0757", force="1", cons="02015R0757-20161216"),
    )
    assert parse_topic_response("mrv", p)[0].candidate_ref == "02015R0757-20250101"


def test_no_consolidations_gives_none_candidate():
    p = payload(binding("32023R2449", force="1"))
    assert parse_topic_response("mrv", p)[0].candidate_ref is None


def test_specs_carry_topic_and_source():
    p = payload(binding("32023R1805", force="1"))
    spec = parse_topic_response("fueleu", p)[0]
    assert spec == DocumentSpec(
        topic="fueleu", source="eurlex", ref="32023R1805", candidate_ref=None
    )


def test_topic_query_embeds_seed():
    q = topic_query("32023R1805")
    assert "resource/celex/32023R1805" in q
    assert "resource_legal_based_on_resource_legal" in q


def test_discover_raises_when_seed_missing_from_results():
    def handler(request):
        return httpx.Response(200, json=payload(binding("32023R2449", force="1")))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DiscoveryError, match="32015R0757"):
            discover(client, "mrv", "32015R0757")


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
        specs = discover(client, "mrv", "32015R0757")
    assert [s.ref for s in specs] == ["32015R0757", "32023R2449"]


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
        specs = discover(client, topic, SEEDS[topic])
        resolved = {f"{topic}:{s.ref}": resolve(client, s).resolved_ref for s in specs}
    expected = {k: v for k, v in EXPECTED_RESOLVED.items() if k.startswith(f"{topic}:")}
    assert resolved == expected
