"""Corpus fetch: version-diff against the previous run, download only what changed."""

import hashlib
import json
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.http import http_client, transient_retry
from app.db.schemas import IngestedDocument, IngestRun
from app.ingestion.discover import SEEDS, DiscoveryError, DocumentSpec, discover
from app.ingestion.enums import DocAction, IngestRunStatus
from app.ingestion.eurlex import ResolutionError, resolve
from app.ingestion.service import complete_ingest_run, create_ingest_run, get_baseline_docs

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
PACE_SECONDS = 1.0


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

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.COMPLETED if self.ok else IngestRunStatus.FAILED

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


def pace() -> None:
    """Space out requests to the upstream host between documents."""
    time.sleep(PACE_SECONDS)


@contextmanager
def open_client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    """Yield the caller's client untouched, or an owned one closed on exit."""
    if client is not None:
        yield client
        return
    with http_client() as owned:
        yield owned


@transient_retry
def download(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def store(data_dir: Path, ref: str, content: bytes) -> tuple[str, int]:
    """Write {ref}.html and return its (sha256, size_bytes)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{ref}.html").write_bytes(content)
    return hashlib.sha256(content).hexdigest(), len(content)


def discover_topics(client: httpx.Client, topics: Sequence[str]) -> list[DocumentSpec]:
    """Discover all topics, deduped by ref (first topic wins), wrapping parse errors."""
    by_ref: dict[str, DocumentSpec] = {}
    for topic in topics:
        try:
            specs = discover(client, topic, SEEDS[topic])
        except (KeyError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"{topic}: malformed SPARQL response: {exc!r}") from exc
        for spec in specs:
            by_ref.setdefault(spec.ref, spec)
    return list(by_ref.values())


def ingest_document(
    client: httpx.Client,
    spec: DocumentSpec,
    prev: IngestedDocument | None,
    run: IngestRun,
    data_dir: Path,
) -> tuple[IngestedDocument, DocAction]:
    """Resolve one act, download it unless unchanged, and build its row."""
    resolution = resolve(client, spec)
    action = classify(prev.resolved_ref if prev else None, resolution.resolved_ref)
    if action is DocAction.UNCHANGED and prev is not None:
        sha256, size_bytes, fetched_at = prev.sha256, prev.size_bytes, prev.fetched_at
    else:
        sha256, size_bytes = store(data_dir, spec.ref, download(client, resolution.url))
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


def ingest_documents(
    client: httpx.Client,
    specs: Sequence[DocumentSpec],
    baseline: Mapping[str, IngestedDocument],
    run: IngestRun,
    data_dir: Path,
    report: RunReport,
) -> list[IngestedDocument]:
    """Ingest each spec in turn, recording its outcome (or its error) on the report."""
    documents = []
    for position, spec in enumerate(specs):
        if position:
            pace()
        try:
            document, action = ingest_document(client, spec, baseline.get(spec.ref), run, data_dir)
        except (ResolutionError, httpx.HTTPError) as exc:
            report.failed[spec.ref] = f"{type(exc).__name__}: {exc}"
            continue
        documents.append(document)
        report.record(action, spec.ref)
    return documents


async def fetch_topics(
    session: AsyncSession,
    topics: Sequence[str],
    data_dir: Path,
    client: httpx.Client | None = None,
) -> RunReport:
    """Fetch the corpus for topics in one ingest run; blocking HTTP is fine here (CLI-only)."""
    run = await create_ingest_run(session)
    report = RunReport(run_id=run.id)
    try:
        with open_client(client) as http:
            specs = discover_topics(http, topics)
            baseline = await get_baseline_docs(session, topics)
            report.dropped = dropped_refs(specs, baseline)
            session.add_all(ingest_documents(http, specs, baseline, run, data_dir, report))
    except (DiscoveryError, httpx.HTTPError):
        await complete_ingest_run(session, run, IngestRunStatus.FAILED)
        raise
    await complete_ingest_run(session, run, report.status)
    return report
