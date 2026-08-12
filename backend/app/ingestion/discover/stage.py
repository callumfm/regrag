"""Discover stage: one SPARQL query per topic seed, deduped, diffed against the last run."""

import json
from collections.abc import Iterable, Sequence

import httpx

from app.core.http import http_retry
from app.ingestion.constants import MAX_DROP_RATIO, MIN_SUSPICIOUS_DROPS, SEEDS
from app.ingestion.discover.models import DiscoveredDocument, DiscoverRunResult
from app.ingestion.discover.select import select_topic_documents
from app.ingestion.discover.sparql import SPARQL_ENDPOINT, collect_candidate_acts, topic_query
from app.ingestion.exceptions import CorpusShrankError, MalformedDiscoveryError


@http_retry
async def discover_topic(
    client: httpx.AsyncClient, topic: str, seed_celex: str
) -> list[DiscoveredDocument]:
    """Run the topic query and select from it; a result set without the seed act is an error."""
    response = await client.get(
        SPARQL_ENDPOINT,
        params={"query": topic_query(seed_celex), "format": "application/sparql-results+json"},
    )
    response.raise_for_status()
    documents = select_topic_documents(topic, collect_candidate_acts(response.json()))
    if not any(document.celex == seed_celex for document in documents):
        raise MalformedDiscoveryError(f"{topic}: seed {seed_celex} missing from discovery results")
    return documents


async def discover_topics(
    client: httpx.AsyncClient, topics: Sequence[str]
) -> list[DiscoveredDocument]:
    """Discover all topics, deduped by celex (first topic wins), wrapping parse errors."""
    by_celex: dict[str, DiscoveredDocument] = {}
    for topic in topics:
        try:
            documents = await discover_topic(client, topic, SEEDS[topic])
        except (KeyError, json.JSONDecodeError) as exc:
            raise MalformedDiscoveryError(f"{topic}: malformed SPARQL response: {exc!r}") from exc
        for document in documents:
            by_celex.setdefault(document.celex, document)
    return list(by_celex.values())


def find_dropped_celexes(
    documents: Sequence[DiscoveredDocument], previous_celexes: Iterable[str]
) -> list[str]:
    """Celexes the previous run held that discovery no longer returns.

    Losing an implausible share is an error: a truncated result set is indistinguishable from a
    mass repeal, so refuse to call it one.
    """
    discovered = {document.celex for document in documents}
    previous = set(previous_celexes)
    dropped = sorted(previous - discovered)
    if len(dropped) >= MIN_SUSPICIOUS_DROPS and len(dropped) > MAX_DROP_RATIO * len(previous):
        raise CorpusShrankError(
            f"discovery lost {len(dropped)} of {len(previous)} documents: {', '.join(dropped)}"
        )
    return dropped


async def discover_corpus(
    client: httpx.AsyncClient, *, topics: Sequence[str], previous_celexes: Iterable[str]
) -> tuple[list[DiscoveredDocument], DiscoverRunResult]:
    """Discover every topic's corpus and diff it against what the previous run held."""
    documents = await discover_topics(client, topics)
    dropped = find_dropped_celexes(documents, previous_celexes)
    result = DiscoverRunResult(dropped=dropped)
    return documents, result
