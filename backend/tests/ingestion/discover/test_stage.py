"""Discover stage: every topic's corpus deduped into one, and what the last run held but it lost."""

import httpx
import pytest

from app.ingestion.discover.stage import discover_topics, find_dropped_celexes
from tests.conftest import MRV_SPARQL, binding, discovered_document, payload

pytestmark = pytest.mark.anyio


async def test_discover_topics_returns_every_topic_corpus(corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, {})

    documents = await discover_topics(client, ["mrv"])

    assert [document.celex for document in documents] == ["32015R0757", "32023R2449"]


async def test_an_act_two_topics_both_return_is_kept_once(corpus_client):
    """First topic wins, so a shared act carries the topic that claimed it first."""
    fueleu = httpx.Response(
        200,
        json=payload(binding("32023R1805", force="1"), binding("32023R2449", force="1")),
    )
    client, _ = corpus_client({"mrv": MRV_SPARQL, "fueleu": fueleu}, {})

    documents = await discover_topics(client, ["mrv", "fueleu"])

    assert [d.celex for d in documents] == ["32015R0757", "32023R2449", "32023R1805"]
    assert {d.celex: d.topic for d in documents}["32023R2449"] == "mrv"


def test_dropped_celexes_are_the_previous_ones_discovery_no_longer_returns():
    found = [discovered_document("32015R0757"), discovered_document("32016R1928")]
    previous = ["32015R0757", "32016R1928", "32014R0666"]
    assert find_dropped_celexes(found, previous) == ["32014R0666"]


def test_nothing_is_dropped_when_every_previous_celex_is_discovered():
    assert find_dropped_celexes([discovered_document("32015R0757")], ["32015R0757"]) == []
