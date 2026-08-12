"""Fetch stage: version-diff against the previous run, download only what changed."""

import httpx
from sqlalchemy.exc import SQLAlchemyError
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
    store: ObjectStore, discovered: DiscoveredDocument, previous: RawDocument | None
) -> Fetched | None:
    """The version and bytes the previous run stored, if the download would land on that version.

    Sparing the request is the point: an unchanged act is the common case, and reading its
    stored bytes both proves they are still there and gives parse what it needs.
    """
    expected = expected_version(discovered)
    if previous is None or previous.resolved_celex != expected.resolved_celex:
        return None
    try:
        content = read_document(store, previous)
    except StorageError:
        return None
    stored = StoredBytes(
        sha256=previous.sha256, size_bytes=previous.size_bytes, fetched_at=previous.fetched_at
    )
    return expected, stored, content


async def _download_new_version(
    client: httpx.AsyncClient, store: ObjectStore, discovered: DiscoveredDocument
) -> Fetched:
    """Download the version EUR-Lex will serve, store its bytes, and stamp the fetch time."""
    resolution, content = await download_fetchable_version(client, discovered)
    sha256, size_bytes = write_document(store, discovered.celex, resolution.resolved_celex, content)
    stored = StoredBytes(sha256=sha256, size_bytes=size_bytes, fetched_at=utc_now())
    return resolution, stored, content


async def _reuse_or_download(
    client: httpx.AsyncClient,
    discovered: DiscoveredDocument,
    *,
    previous: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[FetchedDocument, DocChange]:
    """Reuse the version the last run stored, or download the one discovery now points at."""
    reused = _reuse_stored_version(store, discovered, previous)
    resolution, stored, content = reused or await _download_new_version(client, store, discovered)
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
    client: httpx.AsyncClient,
    discovered: DiscoveredDocument,
    previous: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[FetchedDocument | None, FetchRunResult]:
    """Download one discovered document and record its row, or record why it would not download."""
    result = FetchRunResult()
    try:
        async with session.begin_nested():
            fetched, change = await _reuse_or_download(
                client, discovered, previous=previous, run=run, store=store
            )
            session.add(fetched.document)
            await session.flush()
    except (IngestionError, StorageError, httpx.HTTPError, SQLAlchemyError) as exc:
        result.fail(discovered.celex, exc)
        return None, result
    result.record(change, discovered.celex)
    return fetched, result
