"""Corpus fetch: version-diff against the previous run, download only what changed."""

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_now
from app.core.http import transient_retry
from app.ingestion.enums import DocAction, IngestRunStatus
from app.ingestion.exceptions import DiscoveryError, IngestionError
from app.ingestion.fetch.discover import SEEDS, DocumentSpec, discover
from app.ingestion.fetch.resolve import resolve
from app.ingestion.schemas import IngestedDocument, IngestRun
from app.ingestion.service import (
    complete_ingest_run,
    create_ingest_run,
    get_baseline_docs,
    next_corpus_version,
)

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
        fetched_at = utc_now()
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
    for idx, spec in enumerate(specs):
        if idx > 0:
            pace()
        try:
            document, action = ingest_document(client, spec, baseline.get(spec.ref), run, data_dir)
        except (IngestionError, httpx.HTTPError) as exc:
            report.failed[spec.ref] = f"{type(exc).__name__}: {exc}"
            continue
        documents.append(document)
        report.record(action, spec.ref)
    return documents


async def fetch_topics(
    session: AsyncSession,
    client: httpx.Client,
    topics: Sequence[str],
    data_dir: Path,
) -> RunReport:
    """Fetch the corpus for topics in one ingest run; blocking HTTP is fine here (CLI-only)."""
    run = await create_ingest_run(session)
    report = RunReport(run_id=run.id)
    try:
        specs = discover_topics(client, topics)
        baseline = await get_baseline_docs(session, topics)
        report.dropped = dropped_refs(specs, baseline)
        documents = ingest_documents(client, specs, baseline, run, data_dir, report)
        session.add_all(documents)
    except (DiscoveryError, httpx.HTTPError):
        await complete_ingest_run(session, run, IngestRunStatus.FAILED)
        raise
    version = await next_corpus_version(session, documents) if report.ok else None
    await complete_ingest_run(session, run, report.status, version)
    return report
