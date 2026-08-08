"""Fetch stage: version-diff against the previous run, download only what changed."""

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.http import download, pace
from app.ingestion.constants import PACE_SECONDS
from app.ingestion.enums import DocChange
from app.ingestion.exceptions import EmptyDownloadError, IngestionError
from app.ingestion.fetch.discover import discover_topics, find_dropped_celexes
from app.ingestion.fetch.models import DiscoveredDocument, FetchRunResult
from app.ingestion.fetch.resolve import resolve_version
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.schemas import IngestRun


def _classify(prev_resolved_celex: str | None, resolved_celex: str) -> DocChange:
    if prev_resolved_celex is None:
        return DocChange.NEW
    if prev_resolved_celex != resolved_celex:
        return DocChange.CHANGED
    return DocChange.UNCHANGED


def _store(data_dir: Path, celex: str, content: bytes) -> tuple[str, int]:
    """Write the document's source file and return its (sha256, size_bytes).
    Empty content is refused: it would overwrite the last good copy with nothing.
    """
    if not content:
        raise EmptyDownloadError(f"{celex}: download returned an empty body")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / RawDocument.filename(celex)).write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


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
    data_dir: Path,
) -> tuple[RawDocument, DocChange]:
    """Resolve one act, download it unless unchanged and still on disk, and build its row."""
    resolved = resolve_version(client, spec)
    change = _classify(prev.resolved_celex if prev else None, resolved.resolved_celex)
    if change is DocChange.UNCHANGED and prev is not None and prev.path(data_dir).exists():
        sha256, size_bytes, fetched_at = prev.sha256, prev.size_bytes, prev.fetched_at
    else:
        sha256, size_bytes = _store(data_dir, spec.celex, download(client, resolved.url))
        fetched_at = utc_now()
    document = RawDocument(
        **spec.model_dump(exclude={"candidate_celex"}),
        **resolved.model_dump(),
        run=run,
        sha256=sha256,
        size_bytes=size_bytes,
        fetched_at=fetched_at,
    )
    return document, change


def _download_documents(
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
    for spec in _paced(specs):
        try:
            document, change = _fetch_document(
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
    documents, result = _download_documents(
        client, specs, baseline=baseline, run=run, data_dir=data_dir
    )
    session.add_all(documents)
    await session.flush()
    return documents, result + FetchRunResult(dropped=dropped)
