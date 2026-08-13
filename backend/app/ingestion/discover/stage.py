"""Discover stage: one SPARQL query per topic's base act, deduped into a single corpus."""

from collections.abc import Sequence

import httpx

from app.core.config import config
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.select import select_documents
from app.ingestion.discover.sparql import run_acts_by_topic_query


async def discover_topics(
    client: httpx.AsyncClient, topics: Sequence[str]
) -> list[DiscoveredDocument]:
    """Every topic's corpus, deduped by celex — the first topic to claim an act keeps it."""
    by_celex: dict[str, DiscoveredDocument] = {}
    for topic in topics:
        rows = await run_acts_by_topic_query(client, config.TOPIC_BASE_ACTS[topic])
        for document in select_documents(topic, rows):
            by_celex.setdefault(document.celex, document)
    return list(by_celex.values())
