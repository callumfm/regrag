"""Fetch stage: version-diff against the previous run, download only what changed."""

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.http import download, pace
from app.core.storage import ObjectStore, StorageError
from app.ingestion.constants import MAX_DROP_RATIO, MIN_SUSPICIOUS_DROPS, PACE_SECONDS, SEEDS
from app.ingestion.enums import DocAction
from app.ingestion.exceptions import (
    DiscoveryError,
    EmptyCorpusError,
    EmptyDocumentError,
    IngestionError,
)
from app.ingestion.fetch.discover import discover
from app.ingestion.fetch.models import DiscoveredDocument, FetchRunResult
from app.ingestion.fetch.resolve import resolve_version
from app.ingestion.fetch.schemas import RawDocument, object_key
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.schemas import IngestRun

CARRIED_FIELDS = (
    "source",
    "celex",
    "resolved_celex",
    "topic",
    "url",
    "sha256",
    "size_bytes",
    "fetched_at",
)
"""What a reparse copies forward from the row the previous run recorded."""


def _classify(prev_resolved_celex: str | None, resolved_celex: str) -> DocAction:
    if prev_resolved_celex is None:
        return DocAction.NEW
    if prev_resolved_celex != resolved_celex:
        return DocAction.CHANGED
    return DocAction.UNCHANGED


def _dropped_celexes(
    specs: Sequence[DiscoveredDocument], baseline_celexes: Iterable[str]
) -> list[str]:
    """Baseline celexes discovery no longer returns; losing an implausible share is an error.

    A truncated result set is indistinguishable from a mass repeal, so refuse to call it one.
    """
    discovered = {spec.celex for spec in specs}
    baseline = set(baseline_celexes)
    dropped = sorted(baseline - discovered)
    if len(dropped) >= MIN_SUSPICIOUS_DROPS and len(dropped) > MAX_DROP_RATIO * len(baseline):
        raise DiscoveryError(
            f"discovery lost {len(dropped)} of {len(baseline)} documents: {', '.join(dropped)}"
        )
    return dropped


def _store(store: ObjectStore, celex: str, resolved_celex: str, content: bytes) -> tuple[str, int]:
    """Store the document's bytes and return their (sha256, size_bytes).

    Empty content is refused: it would record a document whose stored bytes are nothing.
    Content already stored under its key is left alone — the same bytes are the same object.
    """
    if not content:
        raise EmptyDocumentError(f"{celex}: download returned an empty body")
    sha256 = hashlib.sha256(content).hexdigest()
    key = object_key(celex, resolved_celex, sha256)
    if not store.exists(key):
        store.put(key, content)
    return sha256, len(content)


def _discover_topics(client: httpx.Client, topics: Sequence[str]) -> list[DiscoveredDocument]:
    """Discover all topics, deduped by celex (first topic wins), wrapping parse errors."""
    by_celex: dict[str, DiscoveredDocument] = {}
    for topic in topics:
        try:
            specs = discover(client, topic, SEEDS[topic])
        except (KeyError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"{topic}: malformed SPARQL response: {exc!r}") from exc
        for spec in specs:
            by_celex.setdefault(spec.celex, spec)
    return list(by_celex.values())


def _paced(specs: Sequence[DiscoveredDocument]) -> Iterator[DiscoveredDocument]:
    """Yield each spec, waiting between them to stay within the source's rate limit."""
    for index, spec in enumerate(specs):
        if index > 0:
            pace(PACE_SECONDS)
        yield spec


def _fetch_document(
    client: httpx.Client,
    spec: DiscoveredDocument,
    *,
    prev: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[RawDocument, DocAction]:
    """Resolve one act, download it unless unchanged and still stored, and build its row."""
    resolution = resolve_version(client, spec)
    action = _classify(prev.resolved_celex if prev else None, resolution.resolved_celex)
    if action is DocAction.UNCHANGED and prev is not None and store.exists(prev.key):
        sha256, size_bytes, fetched_at = prev.sha256, prev.size_bytes, prev.fetched_at
    else:
        sha256, size_bytes = _store(
            store, spec.celex, resolution.resolved_celex, download(client, resolution.url)
        )
        fetched_at = utc_now()
    document = RawDocument(
        **spec.model_dump(exclude={"candidate_celex"}),
        **resolution.model_dump(),
        run=run,
        sha256=sha256,
        size_bytes=size_bytes,
        fetched_at=fetched_at,
    )
    return document, action


def _download_documents(
    client: httpx.Client,
    specs: Sequence[DiscoveredDocument],
    *,
    baseline: Mapping[str, RawDocument],
    run: IngestRun,
    store: ObjectStore,
) -> tuple[list[RawDocument], FetchRunResult]:
    """Fetch every discovered document, recording the ones that would not download."""
    documents: list[RawDocument] = []
    result = FetchRunResult(discovered=[spec.celex for spec in specs])
    for spec in _paced(specs):
        try:
            document, action = _fetch_document(
                client, spec, prev=baseline.get(spec.celex), run=run, store=store
            )
        except (IngestionError, httpx.HTTPError, StorageError) as exc:
            result.fail(spec.celex, exc)
            continue
        documents.append(document)
        result.record(action, spec.celex)
    return documents, result


async def fetch_documents(
    session: AsyncSession,
    *,
    client: httpx.Client,
    topics: Sequence[str],
    store: ObjectStore,
    run: IngestRun,
) -> tuple[list[RawDocument], FetchRunResult]:
    """Discover, resolve and download the corpus for topics, recording a row per document."""
    specs = _discover_topics(client, topics)
    baseline = await get_baseline_docs(session, topics)
    dropped = _dropped_celexes(specs, baseline)
    documents, result = _download_documents(client, specs, baseline=baseline, run=run, store=store)
    session.add_all(documents)
    await session.flush()
    return documents, result + FetchRunResult(dropped=dropped)


async def reuse_documents(
    session: AsyncSession, *, topics: Sequence[str], run: IngestRun
) -> tuple[list[RawDocument], FetchRunResult]:
    """Re-record the corpus as it already stands, so a reparse reaches no network at all.

    The stored bytes are keyed by content hash, so the rows this run copies forward still
    point at exactly what the previous run parsed.
    """
    baseline = await get_baseline_docs(session, topics)
    if not baseline:
        raise EmptyCorpusError(f"nothing fetched yet for: {', '.join(topics)}")
    documents = [
        RawDocument(run=run, **{name: getattr(prev, name) for name in CARRIED_FIELDS})
        for prev in baseline.values()
    ]
    session.add_all(documents)
    await session.flush()
    celexes = sorted(baseline)
    return documents, FetchRunResult(discovered=celexes, unchanged=celexes)
