"""Corpus fetch: version-diff against the previous run, download only what changed."""

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.http import download, pace
from app.ingestion.constants import SEEDS
from app.ingestion.enums import DocAction
from app.ingestion.exceptions import DiscoveryError, IngestionError
from app.ingestion.fetch.discover import discover
from app.ingestion.fetch.models import DiscoveredDocument
from app.ingestion.fetch.resolve import resolve_version
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.models import FetchDelta
from app.ingestion.schemas import IngestRun
from app.ingestion.service import get_baseline_docs


def classify(prev_resolved_ref: str | None, resolved_ref: str) -> DocAction:
    if prev_resolved_ref is None:
        return DocAction.NEW
    if prev_resolved_ref != resolved_ref:
        return DocAction.CHANGED
    return DocAction.UNCHANGED


def dropped_refs(specs: Sequence[DiscoveredDocument], baseline_refs: Iterable[str]) -> list[str]:
    """Baseline refs no longer present in discovery (repealed or out of force)."""
    discovered = {spec.ref for spec in specs}
    return sorted(set(baseline_refs) - discovered)


def store(data_dir: Path, ref: str, content: bytes) -> tuple[str, int]:
    """Write the document's source file and return its (sha256, size_bytes)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / RawDocument.filename(ref)).write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


def discover_topics(client: httpx.Client, topics: Sequence[str]) -> list[DiscoveredDocument]:
    """Discover all topics, deduped by ref (first topic wins), wrapping parse errors."""
    by_ref: dict[str, DiscoveredDocument] = {}
    for topic in topics:
        try:
            specs = discover(client, topic, SEEDS[topic])
        except (KeyError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"{topic}: malformed SPARQL response: {exc!r}") from exc
        for spec in specs:
            by_ref.setdefault(spec.ref, spec)
    return list(by_ref.values())


def fetch_document(
    client: httpx.Client,
    spec: DiscoveredDocument,
    prev: RawDocument | None,
    run: IngestRun,
    data_dir: Path,
) -> tuple[RawDocument, DocAction]:
    """Resolve one act, download it unless unchanged, and build its row."""
    resolution = resolve_version(client, spec)
    action = classify(prev.resolved_ref if prev else None, resolution.resolved_ref)
    if action is DocAction.UNCHANGED and prev is not None:
        sha256, size_bytes, fetched_at = prev.sha256, prev.size_bytes, prev.fetched_at
    else:
        sha256, size_bytes = store(data_dir, spec.ref, download(client, resolution.url))
        fetched_at = utc_now()
    document = RawDocument(
        run=run,
        source=spec.source,
        ref=spec.ref,
        resolved_ref=resolution.resolved_ref,
        topic=spec.topic,
        url=resolution.url,
        sha256=sha256,
        size_bytes=size_bytes,
        fetched_at=fetched_at,
    )
    return document, action


async def fetch_documents(
    session: AsyncSession,
    client: httpx.Client,
    topics: Sequence[str],
    data_dir: Path,
    run: IngestRun,
) -> tuple[list[RawDocument], FetchDelta]:
    """Discover, resolve and download the corpus for topics, recording a row per document."""
    specs = discover_topics(client, topics)
    baseline = await get_baseline_docs(session, topics)
    delta = FetchDelta(
        discovered=[spec.ref for spec in specs], dropped=dropped_refs(specs, baseline)
    )
    documents = []
    for index, spec in enumerate(specs):
        if index > 0:
            pace()
        try:
            document, action = fetch_document(client, spec, baseline.get(spec.ref), run, data_dir)
        except (IngestionError, httpx.HTTPError) as exc:
            delta.failed[spec.ref] = f"{type(exc).__name__}: {exc}"
            continue
        documents.append(document)
        delta.record(action, spec.ref)
    session.add_all(documents)
    await session.flush()
    return documents, delta
