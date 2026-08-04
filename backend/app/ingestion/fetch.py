"""Corpus fetch: version-diff against the previous run, download only what changed."""

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import IngestRunStatus
from app.core.http import http_client
from app.db.schemas import IngestedDocument, IngestRun
from app.ingestion.discover import SEEDS, DiscoveryError, DocumentSpec, discover
from app.ingestion.eurlex import ResolutionError, resolve


class DocAction(StrEnum):
    """How a discovered document compares to the previous run."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


def classify(prev_resolved_ref: str | None, resolved_ref: str) -> DocAction:
    if prev_resolved_ref is None:
        return DocAction.NEW
    if prev_resolved_ref != resolved_ref:
        return DocAction.CHANGED
    return DocAction.UNCHANGED


def dropped_refs(specs: Sequence[DocumentSpec], baseline_refs: Iterable[str]) -> list[str]:
    """Baseline refs no longer present in discovery (repealed or out of force)."""
    discovered = {spec.ref for spec in specs}
    return sorted(set(baseline_refs) - discovered)


@dataclass
class RunReport:
    """Outcome of one fetch run, bucketed for the CLI diff."""

    run_id: int
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    def record(self, action: DocAction, ref: str) -> None:
        bucket = {
            DocAction.NEW: self.new,
            DocAction.CHANGED: self.changed,
            DocAction.UNCHANGED: self.unchanged,
        }
        bucket[action].append(ref)

    def summary(self) -> str:
        lines = [
            f"run {self.run_id}: {len(self.new)} new, {len(self.changed)} changed, "
            f"{len(self.unchanged)} unchanged, {len(self.dropped)} dropped, "
            f"{len(self.failed)} failed"
        ]
        for label, refs in (
            ("new", self.new),
            ("changed", self.changed),
            ("dropped", self.dropped),
        ):
            if refs:
                lines.append(f"  {label}: {', '.join(sorted(refs))}")
        for ref, error in sorted(self.failed.items()):
            lines.append(f"  failed: {ref} ({error})")
        return "\n".join(lines)


DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
RETRY_ATTEMPTS = 3
RETRY_STATUSES = {429, 500, 502, 503, 504}
PACE_SECONDS = 1.0
_sleep = time.sleep


def _retryable(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUSES
    return isinstance(exc, httpx.TransportError)


def with_retry[T](fn: Callable[[], T]) -> T:
    """Retry fn on transient HTTP failures (5xx/429/transport) with exponential backoff."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return fn()
        except httpx.HTTPError as exc:
            if attempt == RETRY_ATTEMPTS - 1 or not _retryable(exc):
                raise
            _sleep(2**attempt)
    raise AssertionError("unreachable")


def download(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def store(data_dir: Path, ref: str, content: bytes) -> tuple[str, int]:
    """Write {ref}.html and return its (sha256, size_bytes)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{ref}.html").write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


async def baseline_documents(
    session: AsyncSession, topics: Sequence[str]
) -> dict[str, IngestedDocument]:
    """Rows of the latest run that recorded documents, filtered to topics, keyed by name."""
    last_run_id = await session.scalar(select(func.max(IngestedDocument.ingest_run_id)))
    if last_run_id is None:
        return {}
    rows = await session.scalars(
        select(IngestedDocument).where(
            IngestedDocument.ingest_run_id == last_run_id,
            IngestedDocument.topic.in_(topics),
        )
    )
    return {row.name: row for row in rows}


def _discover_topics(client: httpx.Client, topics: Sequence[str]) -> list[DocumentSpec]:
    """Discover all topics, deduped by ref (first topic wins), wrapping parse errors."""
    by_ref: dict[str, DocumentSpec] = {}
    for topic in topics:
        seed = SEEDS[topic]
        try:
            specs = with_retry(lambda: discover(client, topic, seed))  # noqa: B023
        except (KeyError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"{topic}: malformed SPARQL response: {exc!r}") from exc
        for spec in specs:
            by_ref.setdefault(spec.ref, spec)
    return list(by_ref.values())


def _ingest_document(
    client: httpx.Client,
    spec: DocumentSpec,
    prev: IngestedDocument | None,
    run: IngestRun,
    data_dir: Path,
) -> tuple[IngestedDocument, DocAction]:
    """Resolve one act, download it unless unchanged, and build its row."""
    resolution = with_retry(lambda: resolve(client, spec))
    action = classify(prev.resolved_ref if prev else None, resolution.resolved_ref)
    if action is DocAction.UNCHANGED and prev is not None:
        sha256, size_bytes, fetched_at = prev.sha256, prev.size_bytes, prev.fetched_at
    else:
        content = with_retry(lambda: download(client, resolution.url))
        sha256, size_bytes = store(data_dir, spec.ref, content)
        fetched_at = datetime.now(UTC)
    document = IngestedDocument(
        run=run,
        name=spec.ref,
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


async def _finalize(session: AsyncSession, run: IngestRun, status: IngestRunStatus) -> None:
    run.status = status
    run.completed_at = datetime.now(UTC)
    await session.commit()


async def fetch_topics(
    session: AsyncSession,
    topics: Sequence[str],
    data_dir: Path,
    client: httpx.Client | None = None,
) -> RunReport:
    """Fetch the corpus for topics in one ingest run; blocking HTTP is fine here (CLI-only)."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    session.add(run)
    await session.commit()
    report = RunReport(run_id=run.id)
    own_client = client is None
    if client is None:
        client = http_client()
    try:
        specs = _discover_topics(client, topics)
        baseline = await baseline_documents(session, topics)
        report.dropped = dropped_refs(specs, baseline)
        for position, spec in enumerate(specs):
            if position:
                _sleep(PACE_SECONDS)
            try:
                document, action = _ingest_document(
                    client, spec, baseline.get(spec.ref), run, data_dir
                )
            except (ResolutionError, httpx.HTTPError) as exc:
                report.failed[spec.ref] = f"{type(exc).__name__}: {exc}"
                continue
            session.add(document)
            report.record(action, spec.ref)
    except (DiscoveryError, httpx.HTTPError):
        await _finalize(session, run, IngestRunStatus.FAILED)
        raise
    finally:
        if own_client:
            client.close()
    status = IngestRunStatus.COMPLETED if report.ok else IngestRunStatus.FAILED
    await _finalize(session, run, status)
    return report
