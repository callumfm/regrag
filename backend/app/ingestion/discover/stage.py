"""Discover stage: one SPARQL query per topic seed, deduped, diffed against the last run."""

import json
from collections.abc import Iterable, Sequence

import httpx

from app.core.config import config
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.select import select_topic_documents
from app.ingestion.discover.sparql import extract_acts, run_topic_query
from app.ingestion.exceptions import CorpusShrankError, MalformedDiscoveryError


def _require_seed(documents: Sequence[DiscoveredDocument], topic: str, seed_celex: str) -> None:
    """The query returns the seed act itself, so a result set without it is malformed."""
    if not any(document.celex == seed_celex for document in documents):
        raise MalformedDiscoveryError(f"{topic}: seed {seed_celex} missing from discovery results")


async def discover_topic(
    client: httpx.AsyncClient, topic: str, seed_celex: str
) -> list[DiscoveredDocument]:
    """One topic's corpus: ask CELLAR, read its rows into acts, select the ones worth fetching."""
    rows = await run_topic_query(client, seed_celex)
    acts = extract_acts(rows)
    documents = select_topic_documents(topic, acts)
    _require_seed(documents, topic, seed_celex)
    return documents


async def discover_topics(
    client: httpx.AsyncClient, topics: Sequence[str]
) -> list[DiscoveredDocument]:
    """Discover all topics, deduped by celex (first topic wins), wrapping parse errors."""
    by_celex: dict[str, DiscoveredDocument] = {}
    for topic in topics:
        try:
            documents = await discover_topic(client, topic, config.SEEDS[topic])
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
    suspicious = len(dropped) >= config.MIN_SUSPICIOUS_DROPS
    if suspicious and len(dropped) > config.MAX_DROP_RATIO * len(previous):
        raise CorpusShrankError(
            f"discovery lost {len(dropped)} of {len(previous)} documents: {', '.join(dropped)}"
        )
    return dropped
