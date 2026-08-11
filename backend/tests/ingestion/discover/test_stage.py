"""Discover stage: the seed guard, the resolved corpus per topic, and the dropped-celex diff."""

from pathlib import Path

import httpx
import pytest

from app.ingestion.constants import SEEDS
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.stage import discover_topic, find_dropped_celexes
from app.ingestion.exceptions import MalformedDiscoveryError
from app.ingestion.fetch.download import download_fetchable_version
from tests.conftest import binding, payload

FIXTURES = Path(__file__).parent.parent / "fetch" / "fixtures"

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

DOC_HTML = (FIXTURES / "doc.html").read_text()
MISSING_HTML_PAGE = (FIXTURES / "missing.html").read_text()
SPARQL_FIXTURES = {topic: (FIXTURES / f"sparql-{topic}.json").read_text() for topic in SEEDS}


def spec(celex, topic="mrv"):
    return DiscoveredDocument(topic=topic, source="eurlex", celex=celex, candidate_celex=None)


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


def test_find_dropped_celexes_returns_baseline_celexes_absent_from_discovery():
    found = [spec("32015R0757"), spec("32016R1928")]
    baseline = ["32015R0757", "32016R1928", "32014R0666"]
    assert find_dropped_celexes(found, baseline) == ["32014R0666"]


def test_find_dropped_celexes_is_empty_when_all_are_discovered():
    assert find_dropped_celexes([spec("32015R0757")], ["32015R0757"]) == []
