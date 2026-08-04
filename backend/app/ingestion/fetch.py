"""Corpus fetch: version-diff against the previous run, download only what changed."""

import hashlib
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.schemas import IngestedDocument
from app.ingestion.discover import DocumentSpec


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
