"""Fetch decision logic, run report, and end-to-end corpus fetch."""

import hashlib
import re

import httpx
import pytest
from sqlalchemy import select

from app.ingestion.enums import DocAction, IngestRunStatus
from app.ingestion.exceptions import DiscoveryError, ParseError
from app.ingestion.fetch import pipeline
from app.ingestion.fetch.discover import SEEDS, DocumentSpec
from app.ingestion.fetch.pipeline import (
    RunReport,
    classify,
    download,
    dropped_refs,
    fetch_topics,
    store,
)
from app.ingestion.schemas import IngestRun
from app.ingestion.service import get_baseline_docs

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


def test_report_ok_and_status_track_failures():
    report = RunReport(run_id=1)
    assert report.ok and report.status is IngestRunStatus.COMPLETED
    report.failed["x"] = "ResolutionError: boom"
    assert not report.ok and report.status is IngestRunStatus.FAILED


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
def paces(monkeypatch):
    """Count pacing delays instead of sleeping through them."""
    calls = []
    monkeypatch.setattr(pipeline, "pace", lambda: calls.append(pipeline.PACE_SECONDS))
    return calls


def one_shot_client(response):
    """Client serving a single canned response to any request."""
    return httpx.Client(transport=httpx.MockTransport(lambda request: response))


def test_download_returns_content_bytes():
    client = one_shot_client(httpx.Response(200, content=b"<html>act</html>"))
    assert download(client, "https://example.eu/doc") == b"<html>act</html>"


def test_download_raises_on_error_status():
    client = one_shot_client(httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        download(client, "https://example.eu/doc")


def test_store_writes_file_and_returns_sha_and_size(tmp_path):
    content = b"<html>act</html>"
    sha256, size = store(tmp_path / "raw", "32023R1805", content)
    assert (tmp_path / "raw" / "32023R1805.html").read_bytes() == content
    assert sha256 == hashlib.sha256(content).hexdigest()
    assert size == len(content)


def binding(celex, force=None, cons=None):
    b = {"c": {"value": celex}}
    if force is not None:
        b["force"] = {"value": force}
    if cons is not None:
        b["cons"] = {"value": cons}
    return b


def payload(*bindings):
    return {"results": {"bindings": list(bindings)}}


def corpus_client(sparql, docs, calls=None):
    """Transport serving SPARQL payloads per topic and HTML responses per celex ref."""
    calls = calls if calls is not None else []

    def handler(request):
        if request.url.host == "publications.europa.eu":
            query = request.url.params["query"]
            for topic, seed in SEEDS.items():
                if seed in query:
                    return sparql[topic]
            raise AssertionError(f"no seed in query: {query[:80]}")
        celex = request.url.params["uri"].removeprefix("CELEX:")
        calls.append(celex)
        return docs[celex]

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


MRV_SPARQL = httpx.Response(
    200, json=payload(binding("32015R0757", force="1"), binding("32023R2449", force="1"))
)


async def test_first_run_ingests_all_as_new(db_session, tmp_path):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
            "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
        },
    )
    report = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    assert sorted(report.new) == ["32015R0757", "32023R2449"]
    assert report.ok
    assert (tmp_path / "32015R0757.html").read_bytes() == b"<html>mrv</html>"
    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.COMPLETED
    assert run.completed_at is not None
    rows = await get_baseline_docs(db_session, ["mrv"])
    assert rows["32023R2449"].name == "32023R2449"
    assert rows["32023R2449"].resolved_ref == "32023R2449"


async def test_unchanged_doc_skips_download_and_carries_sha(db_session, tmp_path):
    docs = {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch_topics(db_session, client, ["mrv"], tmp_path)

    client, calls = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    assert sorted(second.unchanged) == ["32015R0757", "32023R2449"]
    assert calls.count("32015R0757") == 1
    firsts = {r.name: r.sha256 for r in (await get_baseline_docs(db_session, ["mrv"])).values()}
    assert firsts["32015R0757"] == hashlib.sha256(b"<html>mrv</html>").hexdigest()


async def test_new_consolidation_is_changed_and_redownloaded(db_session, tmp_path):
    docs = {
        "32015R0757": httpx.Response(200, content=b"<html>v1</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch_topics(db_session, client, ["mrv"], tmp_path)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    docs = {
        "02015R0757-20250101": httpx.Response(200, content=b"<html>v2</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    }
    client, _ = corpus_client({"mrv": consolidated}, docs)
    report = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    assert report.changed == ["32015R0757"]
    assert (tmp_path / "32015R0757.html").read_bytes() == b"<html>v2</html>"
    rows = await get_baseline_docs(db_session, ["mrv"])
    assert rows["32015R0757"].resolved_ref == "02015R0757-20250101"


async def test_vanished_doc_reported_dropped(db_session, tmp_path):
    docs = {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch_topics(db_session, client, ["mrv"], tmp_path)

    only_seed = httpx.Response(200, json=payload(binding("32015R0757", force="1")))
    client, _ = corpus_client({"mrv": only_seed}, docs)
    report = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    assert report.dropped == ["32023R2449"]
    assert "32023R2449" not in await get_baseline_docs(db_session, ["mrv"])


async def test_per_doc_failure_continues_and_marks_run_failed(db_session, tmp_path):
    docs = {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(400, text="bad"),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    report = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    assert report.new == ["32015R0757"]
    assert "32023R2449" in report.failed
    assert not report.ok
    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None
    assert "32023R2449" not in await get_baseline_docs(db_session, ["mrv"])


async def test_sparql_failure_aborts_and_marks_run_failed(db_session, tmp_path):
    client, _ = corpus_client({"mrv": httpx.Response(500, text="down")}, {})
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_topics(db_session, client, ["mrv"], tmp_path)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None


async def test_completed_run_is_stamped_with_a_dated_corpus_version(db_session, tmp_path):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
            "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
        },
    )
    report = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    run = await db_session.get(IngestRun, report.run_id)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{7}", run.corpus_version)


async def test_unchanged_corpus_keeps_the_previous_corpus_version(db_session, tmp_path):
    docs = {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    first = await fetch_topics(db_session, client, ["mrv"], tmp_path)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] is not None
    assert versions[0] == versions[1]


async def test_changed_document_produces_a_new_corpus_version(db_session, tmp_path):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=b"<html>v1</html>"),
            "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
        },
    )
    first = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    client, _ = corpus_client(
        {"mrv": consolidated},
        {
            "02015R0757-20250101": httpx.Response(200, content=b"<html>v2</html>"),
            "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
        },
    )
    second = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] != versions[1]


async def test_failed_run_is_not_stamped_with_a_corpus_version(db_session, tmp_path):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
            "32023R2449": httpx.Response(400, text="bad"),
        },
    )
    report = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED
    assert run.corpus_version is None


async def test_any_ingestion_error_is_recorded_per_document(db_session, tmp_path, monkeypatch):
    """The per-document loop catches the whole IngestionError family, not just resolution."""
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
            "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
        },
    )

    def unparseable(*args, **kwargs):
        raise ParseError("unrecognised EUR-Lex dialect")

    monkeypatch.setattr(pipeline, "ingest_document", unparseable)
    report = await fetch_topics(db_session, client, ["mrv"], tmp_path)

    assert sorted(report.failed) == ["32015R0757", "32023R2449"]
    assert set(report.failed.values()) == {"ParseError: unrecognised EUR-Lex dialect"}
    assert not report.ok
    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED


async def test_malformed_sparql_payload_raises_discovery_error(db_session, tmp_path):
    client, _ = corpus_client({"mrv": httpx.Response(200, json={"unexpected": True})}, {})
    with pytest.raises(DiscoveryError, match="malformed"):
        await fetch_topics(db_session, client, ["mrv"], tmp_path)


async def test_duplicate_ref_across_topics_ingested_once(db_session, tmp_path):
    shared = binding("32015R0757", force="1")
    sparql = {
        "mrv": httpx.Response(200, json=payload(shared)),
        "fueleu": httpx.Response(200, json=payload(binding("32023R1805", force="1"), shared)),
    }
    docs = {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R1805": httpx.Response(200, content=b"<html>fueleu</html>"),
    }
    client, _ = corpus_client(sparql, docs)
    report = await fetch_topics(db_session, client, ["fueleu", "mrv"], tmp_path)

    assert report.ok
    rows = await get_baseline_docs(db_session, ["fueleu", "mrv"])
    assert rows["32015R0757"].topic == "fueleu"


async def test_paces_between_documents(db_session, tmp_path, paces):
    docs = {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch_topics(db_session, client, ["mrv"], tmp_path)
    assert paces == [pipeline.PACE_SECONDS]
