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
from app.ingestion.fetch.models import FetchedDocument
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.schemas import IngestRun
from app.ingestion.storage import StoredBytesMismatchError, read_document, write_document


def _reuse_previous_version(
    store: ObjectStore,
    discovered: DiscoveredDocument,
    previous: RawDocument | None,
    run: IngestRun,
) -> tuple[RawDocument, bytes] | None:
    """This run's row over the version the previous run stored, if the download would land there.

    Discovery offering the same candidates is what settles that: the version EUR-Lex served
    for them is the one it will serve again, whether that was a candidate or the original act.
    """
    if previous is None or tuple(previous.candidates) != discovered.candidates:
        return None
    try:
        html = read_document(store, previous)
    except (ObjectNotFoundError, StoredBytesMismatchError):
        return None
    raw = RawDocument(
        **discovered.model_dump(),
        run=run,
        resolved_celex=previous.resolved_celex,
        sha256=previous.sha256,
        size_bytes=previous.size_bytes,
        fetched_at=previous.fetched_at,
    )
    return raw, html


async def _download_new_version(
    client: httpx.AsyncClient,
    store: ObjectStore,
    discovered: DiscoveredDocument,
    run: IngestRun,
) -> tuple[RawDocument, bytes]:
    """Download the version EUR-Lex will serve, store its bytes, and stamp the fetch time."""
    resolved_celex, html = await download_fetchable_version(client, discovered)
    sha256, size_bytes = write_document(store, discovered.celex, resolved_celex, html)
    raw = RawDocument(
        **discovered.model_dump(),
        run=run,
        resolved_celex=resolved_celex,
        sha256=sha256,
        size_bytes=size_bytes,
        fetched_at=utc_now(),
    )
    return raw, html


async def fetch_document(
    session: AsyncSession,
    *,
    client: httpx.AsyncClient,
    discovered: DiscoveredDocument,
    previous: RawDocument | None,
    run: IngestRun,
    store: ObjectStore,
) -> FetchedDocument:
    """Record one document's row and hand back its bytes, or say why it would not download."""
    try:
        reused = _reuse_previous_version(store, discovered, previous, run)
        raw, html = reused or await _download_new_version(client, store, discovered, run)
        session.add(raw)
        await session.flush()
    except (IngestionError, StorageError, httpx.HTTPError, SQLAlchemyError) as exc:
        raise DocumentFailed(Stage.FETCH, discovered.celex, exc) from exc
    change = DocChange.between(previous.resolved_celex if previous else None, raw.resolved_celex)
    return FetchedDocument(raw, html, change)
