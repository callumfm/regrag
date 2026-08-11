"""Fetch stage: version-diff against the previous run, download only what changed."""

from collections.abc import Mapping, Sequence

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.storage import ObjectStore, StorageError
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.stage import discover_topics, find_dropped_celexes
from app.ingestion.enums import DocChange
from app.ingestion.exceptions import IngestionError
from app.ingestion.fetch.download import download_fetchable_version, expected_version
from app.ingestion.fetch.models import (
    FetchedDocument,
    FetchRunResult,
    ResolvedVersion,
    StoredBytes,
)
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.schemas import IngestRun
from app.ingestion.storage import read_document, write_document

Fetched = tuple[ResolvedVersion, StoredBytes, bytes]


def _reuse_stored_version(
    store: ObjectStore, spec: DiscoveredDocument, prev: RawDocument | None
) -> Fetched | None:
    """The version and bytes the previous run stored, if the download would land on that version.

    Sparing the request is the point: an unchanged act is the common case, and reading its
    stored bytes both proves they are still there and gives parse what it needs.
    """
    expected = expected_version(spec)
    if prev is None or prev.resolved_celex != expected.resolved_celex:
        return None
    try:
        content = read_document(store, prev)
    except StorageError:
        return None
    stored = StoredBytes(sha256=prev.sha256, size_bytes=prev.size_bytes, fetched_at=prev.fetched_at)
    return expected, stored, content


def _download_new_version(
    client: httpx.Client, store: ObjectStore, spec: DiscoveredDocument
) -> Fetched:
    """Download the version EUR-Lex will serve, store its bytes, and stamp the fetch time."""
    resolution, content = download_fetchable_version(client, spec)
    sha256, size_bytes = write_document(store, spec.celex, resolution.resolved_celex, content)
    stored = StoredBytes(sha256=sha256, size_bytes=size_bytes, fetched_at=utc_now())
    return resolution, stored, content


def _fetch_document(
    client: httpx.Client,
    spec: DiscoveredDocument,
    *,
    prev: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[FetchedDocument, DocChange]:
    """Reuse the version the last run stored, or download the one discovery now points at."""
    resolution, stored, content = _reuse_stored_version(store, spec, prev) or _download_new_version(
        client, store, spec
    )
    change = DocChange.between(prev.resolved_celex if prev else None, resolution.resolved_celex)
    document = RawDocument(
        **spec.model_dump(exclude={"candidate_celex"}),
        **resolution.model_dump(),
        **stored.model_dump(),
        run=run,
    )
    return FetchedDocument(document=document, content=content), change


def _download_documents(
    client: httpx.Client,
    specs: Sequence[DiscoveredDocument],
    *,
    baseline: Mapping[str, RawDocument],
    run: IngestRun,
    store: ObjectStore,
) -> tuple[list[FetchedDocument], FetchRunResult]:
    """Fetch every discovered document, recording the ones that would not download."""
    fetched: list[FetchedDocument] = []
    result = FetchRunResult(discovered=[spec.celex for spec in specs])
    for spec in specs:
        try:
            document, change = _fetch_document(
                client, spec, prev=baseline.get(spec.celex), run=run, store=store
            )
        except (IngestionError, StorageError, httpx.HTTPError) as exc:
            result.fail(spec.celex, exc)
            continue
        fetched.append(document)
        result.record(change, spec.celex)
    return fetched, result


async def fetch_documents(
    session: AsyncSession,
    *,
    client: httpx.Client,
    topics: Sequence[str],
    store: ObjectStore,
    run: IngestRun,
) -> tuple[list[FetchedDocument], FetchRunResult]:
    """Discover, resolve and download the corpus for topics, recording a row per document."""
    specs = discover_topics(client, topics)
    baseline = await get_baseline_docs(session, topics)
    dropped = find_dropped_celexes(specs, baseline)
    fetched, result = _download_documents(client, specs, baseline=baseline, run=run, store=store)
    session.add_all([item.document for item in fetched])
    await session.flush()
    return fetched, result + FetchRunResult(dropped=dropped)
