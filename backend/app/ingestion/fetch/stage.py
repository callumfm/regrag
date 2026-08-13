"""Fetch stage: version-diff against the previous run, download only what changed."""

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.storage import ObjectNotFoundError, ObjectStore, StorageError
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.enums import DocChange, Stage
from app.ingestion.exceptions import DocumentFailed, IngestionError
from app.ingestion.fetch.download import download_fetchable_version
from app.ingestion.fetch.models import FetchedDocument, FetchedVersion
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun
from app.ingestion.storage import StoredBytesMismatchError, read_document, write_document

Fetched = tuple[FetchedVersion, bytes]


def _reuse_stored_version(
    store: ObjectStore, discovered: DiscoveredDocument, previous: RawDocument | None
) -> Fetched | None:
    """The version and bytes the previous run stored, if the download would land on that version.

    Discovery offering the same candidates is what settles that: the version EUR-Lex served
    for them is the one it will serve again, whether that was a candidate or the original act.
    """
    if previous is None or tuple(previous.candidates) != discovered.candidates:
        return None
    try:
        content = read_document(store, previous)
    except (ObjectNotFoundError, StoredBytesMismatchError):
        return None
    version = FetchedVersion(
        resolved_celex=previous.resolved_celex,
        url=previous.url,
        sha256=previous.sha256,
        size_bytes=previous.size_bytes,
        fetched_at=previous.fetched_at,
    )
    return version, content


async def _download_new_version(
    client: httpx.AsyncClient, store: ObjectStore, discovered: DiscoveredDocument
) -> Fetched:
    """Download the version EUR-Lex will serve, store its bytes, and stamp the fetch time."""
    resolution, content = await download_fetchable_version(client, discovered)
    sha256, size_bytes = write_document(store, discovered.celex, resolution.resolved_celex, content)
    version = FetchedVersion(
        **resolution.model_dump(), sha256=sha256, size_bytes=size_bytes, fetched_at=utc_now()
    )
    return version, content


async def fetch_document(
    session: AsyncSession,
    *,
    client: httpx.AsyncClient,
    discovered: DiscoveredDocument,
    previous: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> tuple[FetchedDocument, DocChange]:
    """Download one discovered document and record its row, or say why it would not download."""
    try:
        reused = _reuse_stored_version(store, discovered, previous)
        version, content = reused or await _download_new_version(client, store, discovered)
        change = DocChange.between(
            previous.resolved_celex if previous else None, version.resolved_celex
        )
        document = RawDocument(**discovered.model_dump(), **version.model_dump(), run=run)
        session.add(document)
        await session.flush()
    except (IngestionError, StorageError, httpx.HTTPError, SQLAlchemyError) as exc:
        raise DocumentFailed(Stage.FETCH, discovered.celex, exc) from exc
    return FetchedDocument(document=document, content=content), change
