"""Discover stage: every topic's corpus, deduped into one."""

import httpx
import pytest

from app.ingestion.discover.stage import discover_topics
from tests.conftest import MRV_SPARQL, binding, payload

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
