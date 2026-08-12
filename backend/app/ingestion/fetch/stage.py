"""Fetch stage: version-diff against the previous run, download only what changed."""

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.storage import ObjectStore, StorageError
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.enums import DocChange
from app.ingestion.exceptions import IngestionError
from app.ingestion.fetch.download import download_fetchable_version
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

    Discovery pointing at the same candidate is what settles that: the version EUR-Lex served
    for it is the one it will serve again, whether that was the candidate or the original act.
    """
    if previous is None or previous.candidate_celex != discovered.candidate_celex:
        return None
    try:
        content = read_document(store, previous)
    except StorageError:
        return None
    resolution = ResolvedVersion(resolved_celex=previous.resolved_celex, url=previous.url)
    stored = StoredBytes(
        sha256=previous.sha256, size_bytes=previous.size_bytes, fetched_at=previous.fetched_at
    )
    return resolution, stored, content


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
        **discovered.model_dump(),
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
