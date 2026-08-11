"""Discover stage: the seed guard, the deduped corpus, and the dropped-celex diff."""

import httpx
import pytest

from app.ingestion.discover.stage import discover_corpus, discover_topic, find_dropped_celexes
from app.ingestion.exceptions import MalformedDiscoveryError
from tests.conftest import MRV_SPARQL, binding, discovered_document, payload

pytestmark = pytest.mark.anyio


async def test_discover_raises_when_seed_missing_from_results():
    def handler(request):
        return httpx.Response(200, json=payload(binding("32023R2449", force="1")))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MalformedDiscoveryError, match="32015R0757"):
            await discover_topic(client, "mrv", "32015R0757")


async def test_discover_returns_the_documents_the_query_selected():
    def handler(request):
        assert request.url.params["format"] == "application/sparql-results+json"
        return httpx.Response(
            200,
            json=payload(
                binding("32015R0757", force="1", cons="02015R0757-20250101"),
                binding("32023R2449", force="1"),
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        discovered = await discover_topic(client, "mrv", "32015R0757")
    assert [d.celex for d in discovered] == ["32015R0757", "32023R2449"]


def test_find_dropped_celexes_returns_previous_celexes_absent_from_discovery():
    found = [discovered_document("32015R0757"), discovered_document("32016R1928")]
    previous = ["32015R0757", "32016R1928", "32014R0666"]
    assert find_dropped_celexes(found, previous) == ["32014R0666"]


def test_find_dropped_celexes_is_empty_when_all_are_discovered():
    assert find_dropped_celexes([discovered_document("32015R0757")], ["32015R0757"]) == []


async def test_discover_corpus_reports_what_it_found_and_what_the_previous_run_lost(corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, {})

    documents, result = await discover_corpus(
        client, topics=["mrv"], previous_celexes=["32015R0757", "31999R0001"]
    )

    assert result.dropped == ["31999R0001"]
    assert result.ok
