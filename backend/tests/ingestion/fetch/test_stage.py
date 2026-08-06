"""Fetch decision logic, download and store, and the fetch stage in isolation."""

import hashlib

import httpx
import pytest

from app.ingestion.constants import PACE_SECONDS
from app.ingestion.enums import DocAction
from app.ingestion.exceptions import ParseError
from app.ingestion.fetch import stage
from app.ingestion.fetch.models import DiscoveredDocument, FetchRunResult
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.fetch.stage import _classify, _dropped_refs, _store, fetch_documents
from app.ingestion.service import create_ingest_run
from tests.conftest import MRV_SPARQL, binding, payload

pytestmark = pytest.mark.anyio


def spec(ref, topic="mrv"):
    return DiscoveredDocument(topic=topic, source="eurlex", ref=ref, candidate_ref=None)


def test_classify_no_baseline_is_new():
    assert _classify(None, "32023R2449") is DocAction.NEW


def test_classify_differing_resolved_ref_is_changed():
    assert _classify("02015R0757-20240101", "02015R0757-20250101") is DocAction.CHANGED


def test_classify_same_resolved_ref_is_unchanged():
    assert _classify("02015R0757-20250101", "02015R0757-20250101") is DocAction.UNCHANGED


def test_dropped_refs_are_baseline_refs_absent_from_discovery():
    specs = [spec("32015R0757"), spec("32016R1928")]
    baseline = ["32015R0757", "32016R1928", "32014R0666"]
    assert _dropped_refs(specs, baseline) == ["32014R0666"]


def test_dropped_refs_empty_when_all_discovered():
    assert _dropped_refs([spec("32015R0757")], ["32015R0757"]) == []


def test_store_writes_file_and_returns_sha_and_size(tmp_path):
    content = b"<html>act</html>"
    sha256, size = _store(tmp_path / "raw", "32023R1805", content)
    assert (tmp_path / "raw" / "32023R1805.html").read_bytes() == content
    assert sha256 == hashlib.sha256(content).hexdigest()
    assert size == len(content)


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-ref responses overridable."""
    return {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    } | (overrides or {})


async def fetch(db_session, client, topics, data_dir) -> tuple[FetchRunResult, list[RawDocument]]:
    """Drive the fetch stage alone, with the run the orchestrator would supply."""
    run = await create_ingest_run(db_session)
    documents, result = await fetch_documents(
        db_session, client=client, topics=topics, data_dir=data_dir, run=run
    )
    return result, documents


async def test_first_run_ingests_all_as_new(db_session, tmp_path, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert sorted(report.new) == ["32015R0757", "32023R2449"]
    assert report.ok
    assert (tmp_path / "32015R0757.html").read_bytes() == b"<html>mrv</html>"
    rows = await get_baseline_docs(db_session, ["mrv"])
    assert rows["32023R2449"].ref == "32023R2449"
    assert rows["32023R2449"].resolved_ref == "32023R2449"


async def test_unchanged_doc_skips_download_and_carries_sha(db_session, tmp_path, corpus_client):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], tmp_path)

    client, calls = corpus_client({"mrv": MRV_SPARQL}, docs)
    second, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert sorted(second.unchanged) == ["32015R0757", "32023R2449"]
    assert calls.count("32015R0757") == 1
    firsts = {r.ref: r.sha256 for r in (await get_baseline_docs(db_session, ["mrv"])).values()}
    assert firsts["32015R0757"] == hashlib.sha256(b"<html>mrv</html>").hexdigest()


async def test_new_consolidation_is_changed_and_redownloaded(db_session, tmp_path, corpus_client):
    docs = mrv_docs({"32015R0757": httpx.Response(200, content=b"<html>v1</html>")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], tmp_path)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    docs = mrv_docs({"02015R0757-20250101": httpx.Response(200, content=b"<html>v2</html>")})
    client, _ = corpus_client({"mrv": consolidated}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert report.changed == ["32015R0757"]
    assert (tmp_path / "32015R0757.html").read_bytes() == b"<html>v2</html>"
    rows = await get_baseline_docs(db_session, ["mrv"])
    assert rows["32015R0757"].resolved_ref == "02015R0757-20250101"


async def test_vanished_doc_reported_dropped(db_session, tmp_path, corpus_client):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], tmp_path)

    only_seed = httpx.Response(200, json=payload(binding("32015R0757", force="1")))
    client, _ = corpus_client({"mrv": only_seed}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert report.dropped == ["32023R2449"]
    assert "32023R2449" not in await get_baseline_docs(db_session, ["mrv"])


async def test_per_doc_failure_continues_and_is_recorded(db_session, tmp_path, corpus_client):
    docs = mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert report.new == ["32015R0757"]
    assert "32023R2449" in report.failed
    assert not report.ok
    assert "32023R2449" not in await get_baseline_docs(db_session, ["mrv"])


async def test_any_ingestion_error_is_recorded_per_document(
    db_session, tmp_path, corpus_client, monkeypatch
):
    """The per-document loop catches the whole IngestionError family, not just resolution."""
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())

    def unparseable(*args, **kwargs):
        raise ParseError("unrecognised EUR-Lex dialect")

    monkeypatch.setattr(stage, "_fetch_document", unparseable)
    report, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert sorted(report.failed) == ["32015R0757", "32023R2449"]
    assert set(report.failed.values()) == {"ParseError: unrecognised EUR-Lex dialect"}
    assert not report.ok


async def test_duplicate_ref_across_topics_ingested_once(db_session, tmp_path, corpus_client):
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
    report, _ = await fetch(db_session, client, ["fueleu", "mrv"], tmp_path)

    assert report.ok
    rows = await get_baseline_docs(db_session, ["fueleu", "mrv"])
    assert rows["32015R0757"].topic == "fueleu"


async def test_paces_between_documents(db_session, tmp_path, corpus_client, paces):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await fetch(db_session, client, ["mrv"], tmp_path)
    assert paces == [PACE_SECONDS]
