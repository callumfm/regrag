"""Fetch stage: version-diff against the previous run, download only what changed."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.http import download
from app.ingestion.enums import DocChange
from app.ingestion.exceptions import IngestionError
from app.ingestion.fetch.discover import discover_topics, find_dropped_celexes
from app.ingestion.fetch.models import (
    DiscoveredDocument,
    FetchRunResult,
    ResolvedVersion,
    StoredBytes,
)
from app.ingestion.fetch.resolve import resolve_version
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.fetch.storage import document_exists, write_document
from app.ingestion.schemas import IngestRun


def reuse_stored_bytes(
    data_dir: Path, prev: RawDocument | None, change: DocChange
) -> StoredBytes | None:
    """The previous run's bytes, if this act is unchanged and its file is still stored."""
    if change is not DocChange.UNCHANGED or prev is None or not document_exists(data_dir, prev):
        return None
    return StoredBytes(sha256=prev.sha256, size_bytes=prev.size_bytes, fetched_at=prev.fetched_at)


def download_and_store(
    client: httpx.Client,
    data_dir: Path,
    *,
    spec: DiscoveredDocument,
    resolution: ResolvedVersion,
) -> StoredBytes:
    """Download the act's HTML, store it, and stamp the fetch time."""
    sha256, size_bytes = write_document(data_dir, spec.celex, download(client, resolution.url))
    return StoredBytes(sha256=sha256, size_bytes=size_bytes, fetched_at=utc_now())


def fetch_document(
    client: httpx.Client,
    spec: DiscoveredDocument,
    *,
    prev: RawDocument | None,
    run: IngestRun,
    data_dir: Path,
) -> tuple[RawDocument, DocChange]:
    """Resolve one act, download it unless unchanged and still stored, and build its row."""
    resolution = resolve_version(client, spec)
    change = DocChange.between(prev.resolved_celex if prev else None, resolution.resolved_celex)
    stored = reuse_stored_bytes(data_dir, prev, change) or download_and_store(
        client, data_dir, spec=spec, resolution=resolution
    )
    document = RawDocument(
        **spec.model_dump(exclude={"candidate_celex"}),
        **resolution.model_dump(),
        **stored.model_dump(),
        run=run,
    )
    return document, change


def download_documents(
    client: httpx.Client,
    specs: Sequence[DiscoveredDocument],
    *,
    baseline: Mapping[str, RawDocument],
    run: IngestRun,
    data_dir: Path,
) -> tuple[list[RawDocument], FetchRunResult]:
    """Fetch every discovered document, recording the ones that would not download."""
    documents: list[RawDocument] = []
    result = FetchRunResult(discovered=[spec.celex for spec in specs])
    for spec in specs:
        try:
            document, change = fetch_document(
                client, spec, prev=baseline.get(spec.celex), run=run, data_dir=data_dir
            )
        except (IngestionError, httpx.HTTPError, OSError) as exc:
            result.fail(spec.celex, exc)
            continue
        documents.append(document)
        result.record(change, spec.celex)
    return documents, result


async def fetch_documents(
    session: AsyncSession,
    *,
    client: httpx.Client,
    topics: Sequence[str],
    data_dir: Path,
    run: IngestRun,
) -> tuple[list[RawDocument], FetchRunResult]:
    """Discover, resolve and download the corpus for topics, recording a row per document."""
    specs = discover_topics(client, topics)
    baseline = await get_baseline_docs(session, topics)
    dropped = find_dropped_celexes(specs, baseline)
    documents, result = download_documents(
        client, specs, baseline=baseline, run=run, data_dir=data_dir
    )
    session.add_all(documents)
    await session.flush()
    return documents, result + FetchRunResult(dropped=dropped)
