"""Fetch stage: version-diff against the previous run, download only what changed."""

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.storage import ObjectStore, StorageError
from app.ingestion.discover.models import DiscoveredDocument
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


def _reuse_or_download(
    client: httpx.Client,
    discovered: DiscoveredDocument,
    *,
    previous: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[FetchedDocument, DocChange]:
    """Reuse the version the last run stored, or download the one discovery now points at."""
    resolution, stored, content = _reuse_stored_version(
        store, discovered, previous
    ) or _download_new_version(client, store, discovered)
    change = DocChange.between(
        previous.resolved_celex if previous else None, resolution.resolved_celex
    )
    document = RawDocument(
        **discovered.model_dump(exclude={"candidate_celex"}),
        **resolution.model_dump(),
        **stored.model_dump(),
        run=run,
    )
    return FetchedDocument(document=document, content=content), change


async def fetch_document(
    session: AsyncSession,
    *,
    client: httpx.Client,
    discovered: DiscoveredDocument,
    previous: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[FetchedDocument | None, FetchRunResult]:
    """Download one discovered document and record its row, or record why it would not download."""
    result = FetchRunResult()
    try:
        fetched, change = _reuse_or_download(
            client, discovered, previous=previous, run=run, store=store
        )
    except (IngestionError, StorageError, httpx.HTTPError) as exc:
        result.fail(discovered.celex, exc)
        return None, result
    session.add(fetched.document)
    await session.flush()
    result.record(change, discovered.celex)
    return fetched, result
