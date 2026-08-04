"""Fetch decision logic: version-ref classification, drop detection, run report."""

import hashlib
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import IngestRunStatus
from app.db.schemas import IngestedDocument, IngestRun
from app.ingestion import fetch
from app.ingestion.discover import DocumentSpec
from app.ingestion.fetch import (
    DocAction,
    RunReport,
    baseline_documents,
    classify,
    download,
    dropped_refs,
    store,
    with_retry,
)

pytestmark = pytest.mark.anyio


def spec(ref, topic="mrv"):
    return DocumentSpec(topic=topic, source="eurlex", ref=ref, candidate_ref=None)


def test_classify_no_baseline_is_new():
    assert classify(None, "32023R2449") is DocAction.NEW


def test_classify_differing_resolved_ref_is_changed():
    assert classify("02015R0757-20240101", "02015R0757-20250101") is DocAction.CHANGED


def test_classify_same_resolved_ref_is_unchanged():
    assert classify("02015R0757-20250101", "02015R0757-20250101") is DocAction.UNCHANGED


def test_dropped_refs_are_baseline_refs_absent_from_discovery():
    specs = [spec("32015R0757"), spec("32016R1928")]
    baseline = ["32015R0757", "32016R1928", "32014R0666"]
    assert dropped_refs(specs, baseline) == ["32014R0666"]


def test_dropped_refs_empty_when_all_discovered():
    assert dropped_refs([spec("32015R0757")], ["32015R0757"]) == []


def test_report_record_routes_to_buckets():
    report = RunReport(run_id=1)
    report.record(DocAction.NEW, "a")
    report.record(DocAction.CHANGED, "b")
    report.record(DocAction.UNCHANGED, "c")
    assert (report.new, report.changed, report.unchanged) == (["a"], ["b"], ["c"])


def test_report_ok_iff_no_failures():
    report = RunReport(run_id=1)
    assert report.ok
    report.failed["x"] = "ResolutionError: boom"
    assert not report.ok


def test_summary_counts_and_lists_non_empty_buckets():
    report = RunReport(run_id=7)
    report.record(DocAction.NEW, "32026R0394")
    report.record(DocAction.UNCHANGED, "32015R0757")
    report.dropped.append("32014R0666")
    report.failed["32023R2917"] = "ResolutionError: no fetchable HTML"
    text = report.summary()
    assert "run 7: 1 new, 0 changed, 1 unchanged, 1 dropped, 1 failed" in text
    assert "new: 32026R0394" in text
    assert "dropped: 32014R0666" in text
    assert "failed: 32023R2917 (ResolutionError: no fetchable HTML)" in text
    assert "changed:" not in text


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(fetch, "_sleep", sleeps.append)
    return sleeps


def flaky_client(responses):
    """Client whose handler pops one queued response per request."""
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return responses.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def _get(client, url="https://example.eu/doc"):
    response = client.get(url)
    response.raise_for_status()
    return response


def test_with_retry_recovers_from_retryable_status(no_sleep):
    client, calls = flaky_client([httpx.Response(503), httpx.Response(200, text="ok")])
    assert with_retry(lambda: _get(client)).text == "ok"
    assert len(calls) == 2
    assert no_sleep == [1]


def test_with_retry_gives_up_after_three_attempts(no_sleep):
    client, calls = flaky_client([httpx.Response(503)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        with_retry(lambda: _get(client))
    assert len(calls) == 3
    assert no_sleep == [1, 2]


def test_with_retry_does_not_retry_client_errors():
    client, calls = flaky_client([httpx.Response(404)])
    with pytest.raises(httpx.HTTPStatusError):
        with_retry(lambda: _get(client))
    assert len(calls) == 1


def test_with_retry_recovers_from_transport_error(no_sleep):
    responses = [httpx.ConnectError("refused"), httpx.Response(200, text="ok")]

    def handler(request):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert with_retry(lambda: _get(client)).text == "ok"


def test_download_returns_content_bytes():
    client, _ = flaky_client([httpx.Response(200, content=b"<html>act</html>")])
    assert download(client, "https://example.eu/doc") == b"<html>act</html>"


def test_download_raises_on_error_status():
    client, _ = flaky_client([httpx.Response(500)])
    with pytest.raises(httpx.HTTPStatusError):
        download(client, "https://example.eu/doc")


def test_store_writes_file_and_returns_sha_and_size(tmp_path):
    content = b"<html>act</html>"
    sha256, size = store(tmp_path / "raw", "32023R1805", content)
    assert (tmp_path / "raw" / "32023R1805.html").read_bytes() == content
    assert sha256 == hashlib.sha256(content).hexdigest()
    assert size == len(content)


def make_row(run, ref, topic, resolved_ref=None, sha256="a" * 64):
    return IngestedDocument(
        run=run,
        name=ref,
        source="eurlex",
        ref=ref,
        resolved_ref=resolved_ref or ref,
        topic=topic,
        url=f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{ref}",
        sha256=sha256,
        size_bytes=100,
        fetched_at=datetime.now(UTC),
    )


async def test_baseline_empty_when_no_prior_runs(db_session: AsyncSession):
    assert await baseline_documents(db_session, ["mrv"]) == {}


async def test_baseline_is_latest_run_with_rows_filtered_to_topics(
    db_session: AsyncSession,
):
    old = IngestRun(status=IngestRunStatus.COMPLETED)
    latest = IngestRun(status=IngestRunStatus.FAILED)
    db_session.add_all(
        [
            make_row(old, "32014R0666", "mrv"),
            make_row(latest, "32015R0757", "mrv", resolved_ref="02015R0757-20250101"),
            make_row(latest, "32023R1805", "fueleu"),
        ]
    )
    await db_session.flush()

    baseline = await baseline_documents(db_session, ["mrv"])
    assert set(baseline) == {"32015R0757"}
    assert baseline["32015R0757"].resolved_ref == "02015R0757-20250101"


async def test_baseline_skips_newer_run_without_rows(db_session: AsyncSession):
    with_rows = IngestRun(status=IngestRunStatus.COMPLETED)
    db_session.add(make_row(with_rows, "32015R0757", "mrv"))
    db_session.add(IngestRun(status=IngestRunStatus.FAILED))
    await db_session.flush()

    baseline = await baseline_documents(db_session, ["mrv"])
    assert set(baseline) == {"32015R0757"}
