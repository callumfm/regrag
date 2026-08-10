"""Reusing or downloading a document's version, and the fetch stage in isolation."""

import hashlib

import httpx
import pytest

from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import ParseError
from app.ingestion.fetch import stage
from app.ingestion.fetch.models import DiscoveredDocument, FetchRunResult
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.fetch.stage import _reuse_stored_version, fetch_documents
from app.ingestion.schemas import IngestRun
from app.ingestion.service import create_ingest_run
from app.ingestion.storage import document_filename, write_document
from tests.conftest import MRV_SPARQL, binding, payload

pytestmark = pytest.mark.anyio


def spec(celex, topic="mrv", candidate=None):
    return DiscoveredDocument(topic=topic, source="eurlex", celex=celex, candidate_celex=candidate)


def stored(make_document, tmp_path, celex="32023R1805"):
    """A previous run's row whose bytes are still on disk."""
    document = make_document(IngestRun(status=IngestRunStatus.COMPLETED), celex=celex)
    write_document(tmp_path, celex, b"<html>act</html>")
    return document


def test_stored_version_is_reused_when_discovery_still_points_at_it(tmp_path, make_document):
    prev = stored(make_document, tmp_path)
    reused = _reuse_stored_version(tmp_path, spec("32023R1805"), prev)
    assert reused is not None
    resolution, bytes_ = reused
    assert resolution.resolved_celex == prev.resolved_celex
    assert (bytes_.sha256, bytes_.size_bytes, bytes_.fetched_at) == (
        prev.sha256,
        prev.size_bytes,
        prev.fetched_at,
    )


def test_a_newly_discovered_consolidation_is_not_reused(tmp_path, make_document):
    """Discovery pointing somewhere new is exactly the case that has to hit the network."""
    prev = stored(make_document, tmp_path)
    newer = spec("32023R1805", candidate="02023R1805-20250101")
    assert _reuse_stored_version(tmp_path, newer, prev) is None


def test_stored_version_no_longer_on_disk_is_not_reused(tmp_path, make_document):
    prev = stored(make_document, tmp_path)
    (tmp_path / document_filename(prev.celex)).unlink()
    assert _reuse_stored_version(tmp_path, spec("32023R1805"), prev) is None


def test_a_document_with_no_previous_run_has_nothing_to_reuse(tmp_path):
    assert _reuse_stored_version(tmp_path, spec("32023R1805"), None) is None


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-celex responses overridable."""
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
    assert rows["32023R2449"].celex == "32023R2449"
    assert rows["32023R2449"].resolved_celex == "32023R2449"


async def test_unchanged_run_makes_no_html_requests_and_carries_sha(
    db_session, tmp_path, corpus_client
):
    """Steady state: discovery points at the versions already stored, so nothing is downloaded."""
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], tmp_path)

    client, calls = corpus_client({"mrv": MRV_SPARQL}, docs)
    second, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert sorted(second.unchanged) == ["32015R0757", "32023R2449"]
    assert calls == []
    firsts = {r.celex: r.sha256 for r in (await get_baseline_docs(db_session, ["mrv"])).values()}
    assert firsts["32015R0757"] == hashlib.sha256(b"<html>mrv</html>").hexdigest()


async def test_new_consolidation_is_changed_and_redownloaded(db_session, tmp_path, corpus_client):
    """One request, not two: the download hands back the bytes it already pulled."""
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
    client, calls = corpus_client({"mrv": consolidated}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert report.changed == ["32015R0757"]
    assert calls == ["02015R0757-20250101"]
    assert (tmp_path / "32015R0757.html").read_bytes() == b"<html>v2</html>"
    rows = await get_baseline_docs(db_session, ["mrv"])
    assert rows["32015R0757"].resolved_celex == "02015R0757-20250101"


async def test_still_rendering_doc_fails_without_destroying_its_raw_file(
    db_session, tmp_path, corpus_client
):
    """The regression: a 202 used to be stored as an empty file, wiping the last good copy.

    A new consolidation is what forces the download; an unchanged act is never requested at all.
    """
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await fetch(db_session, client, ["mrv"], tmp_path)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    rendering = mrv_docs({"02015R0757-20250101": httpx.Response(202, content=b"")})
    client, _ = corpus_client({"mrv": consolidated}, rendering)
    report, _ = await fetch(db_session, client, ["mrv"], tmp_path)

    assert "32015R0757" in report.failed
    assert not report.ok
    assert (tmp_path / "32015R0757.html").read_bytes() == b"<html>mrv</html>"
    assert report.unchanged == ["32023R2449"]


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


async def test_duplicate_celex_across_topics_ingested_once(db_session, tmp_path, corpus_client):
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
