"""Fetch decision logic, download and store, and the fetch stage in isolation."""

import hashlib

import httpx
import pytest

from app.core.storage import LocalObjectStore, StorageError
from app.ingestion.constants import PACE_SECONDS
from app.ingestion.enums import DocAction
from app.ingestion.exceptions import EmptyCorpusError, EmptyDocumentError, ParseError
from app.ingestion.fetch import stage
from app.ingestion.fetch.models import DiscoveredDocument, FetchRunResult
from app.ingestion.fetch.schemas import RawDocument, object_key
from app.ingestion.fetch.service import get_baseline_docs
from app.ingestion.fetch.stage import (
    _classify,
    _dropped_celexes,
    _store,
    fetch_documents,
    reuse_documents,
)
from app.ingestion.service import create_ingest_run
from tests.conftest import MRV_SPARQL, binding, payload

pytestmark = pytest.mark.anyio


class CountingStore(LocalObjectStore):
    """A local store that records every key it was actually asked to write."""

    def __init__(self, root):
        super().__init__(root)
        self.puts: list[str] = []

    def put(self, key: str, content: bytes) -> None:
        self.puts.append(key)
        super().put(key, content)


def spec(celex, topic="mrv"):
    return DiscoveredDocument(topic=topic, source="eurlex", celex=celex, candidate_celex=None)


async def stored(db_session, store, celex, topics=("mrv",)) -> bytes:
    """The bytes held for a celex, reached through the row that recorded them."""
    rows = await get_baseline_docs(db_session, list(topics))
    return store.get(rows[celex].key)


def test_classify_no_baseline_is_new():
    assert _classify(None, "32023R2449") is DocAction.NEW


def test_classify_differing_resolved_celex_is_changed():
    assert _classify("02015R0757-20240101", "02015R0757-20250101") is DocAction.CHANGED


def test_classify_same_resolved_celex_is_unchanged():
    assert _classify("02015R0757-20250101", "02015R0757-20250101") is DocAction.UNCHANGED


def test_dropped_celexes_are_baseline_celexes_absent_from_discovery():
    specs = [spec("32015R0757"), spec("32016R1928")]
    baseline = ["32015R0757", "32016R1928", "32014R0666"]
    assert _dropped_celexes(specs, baseline) == ["32014R0666"]


def test_dropped_celexes_empty_when_all_discovered():
    assert _dropped_celexes([spec("32015R0757")], ["32015R0757"]) == []


def test_store_writes_the_object_and_returns_sha_and_size(store):
    content = b"<html>act</html>"
    sha256, size = _store(store, "32023R1805", "32023R1805", content)
    assert store.get(object_key("32023R1805", "32023R1805", sha256)) == content
    assert sha256 == hashlib.sha256(content).hexdigest()
    assert size == len(content)


def test_store_refuses_empty_content(store):
    with pytest.raises(EmptyDocumentError, match="32023R2917"):
        _store(store, "32023R2917", "32023R2917", b"")


def test_store_leaves_the_previous_version_intact_when_content_is_empty(store):
    """An empty body must not destroy the last good copy of the document."""
    sha256, _ = _store(store, "32023R2917", "32023R2917", b"<html>act</html>")
    with pytest.raises(EmptyDocumentError):
        _store(store, "32023R2917", "32023R2917", b"")
    assert store.get(object_key("32023R2917", "32023R2917", sha256)) == b"<html>act</html>"


def test_storing_the_same_content_twice_writes_the_object_once(tmp_path):
    """Same bytes, same key: the second store is a no-op, not a rewrite."""
    store = CountingStore(tmp_path / "raw")
    _store(store, "32023R1805", "32023R1805", b"<html>act</html>")
    _store(store, "32023R1805", "32023R1805", b"<html>act</html>")
    assert len(store.puts) == 1


def test_a_new_version_is_stored_beside_the_bytes_the_last_parse_used(store):
    """A changed document must not overwrite what an earlier parse ran against."""
    first, _ = _store(store, "32015R0757", "32015R0757", b"<html>v1</html>")
    second, _ = _store(store, "32015R0757", "02015R0757-20250101", b"<html>v2</html>")
    assert store.get(object_key("32015R0757", "32015R0757", first)) == b"<html>v1</html>"
    assert store.get(object_key("32015R0757", "02015R0757-20250101", second)) == b"<html>v2</html>"


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-celex responses overridable."""
    return {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    } | (overrides or {})


async def fetch(db_session, client, topics, store) -> tuple[FetchRunResult, list[RawDocument]]:
    """Drive the fetch stage alone, with the run the orchestrator would supply."""
    run = await create_ingest_run(db_session)
    documents, result = await fetch_documents(
        db_session, client=client, topics=topics, store=store, run=run
    )
    return result, documents


async def test_first_run_ingests_all_as_new(db_session, store, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert sorted(report.new) == ["32015R0757", "32023R2449"]
    assert report.ok
    assert await stored(db_session, store, "32015R0757") == b"<html>mrv</html>"
    rows = await get_baseline_docs(db_session, ["mrv"])
    assert rows["32023R2449"].celex == "32023R2449"
    assert rows["32023R2449"].resolved_celex == "32023R2449"


async def test_unchanged_doc_skips_download_and_carries_sha(db_session, store, corpus_client):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], store)

    client, calls = corpus_client({"mrv": MRV_SPARQL}, docs)
    second, _ = await fetch(db_session, client, ["mrv"], store)

    assert sorted(second.unchanged) == ["32015R0757", "32023R2449"]
    assert calls.count("32015R0757") == 1
    firsts = {r.celex: r.sha256 for r in (await get_baseline_docs(db_session, ["mrv"])).values()}
    assert firsts["32015R0757"] == hashlib.sha256(b"<html>mrv</html>").hexdigest()


async def test_unchanged_doc_is_redownloaded_when_its_object_is_gone(
    db_session, tmp_path, corpus_client
):
    """Storage is the source of truth for the bytes; a row alone is not enough to skip a fetch."""
    store = CountingStore(tmp_path / "raw")
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], store)
    key = (await get_baseline_docs(db_session, ["mrv"]))["32015R0757"].key
    (store.root / key).unlink()

    client, calls = corpus_client({"mrv": MRV_SPARQL}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert sorted(report.unchanged) == ["32015R0757", "32023R2449"]
    assert calls.count("32015R0757") == 2, "resolved, then downloaded again"
    assert calls.count("32023R2449") == 1, "resolved only; its object is still there"
    assert store.get(key) == b"<html>mrv</html>"


async def test_new_consolidation_is_changed_and_redownloaded(db_session, store, corpus_client):
    docs = mrv_docs({"32015R0757": httpx.Response(200, content=b"<html>v1</html>")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], store)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    docs = mrv_docs({"02015R0757-20250101": httpx.Response(200, content=b"<html>v2</html>")})
    client, _ = corpus_client({"mrv": consolidated}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert report.changed == ["32015R0757"]
    assert await stored(db_session, store, "32015R0757") == b"<html>v2</html>"
    rows = await get_baseline_docs(db_session, ["mrv"])
    assert rows["32015R0757"].resolved_celex == "02015R0757-20250101"


async def test_a_changed_document_keeps_the_bytes_the_last_parse_ran_against(
    db_session, store, corpus_client
):
    """Telling a parser bug from a source change needs both versions to still be readable."""
    docs = mrv_docs({"32015R0757": httpx.Response(200, content=b"<html>v1</html>")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    _, first = await fetch(db_session, client, ["mrv"], store)
    before = {document.celex: document.key for document in first}

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    docs = mrv_docs({"02015R0757-20250101": httpx.Response(200, content=b"<html>v2</html>")})
    client, _ = corpus_client({"mrv": consolidated}, docs)
    await fetch(db_session, client, ["mrv"], store)

    assert store.get(before["32015R0757"]) == b"<html>v1</html>"
    assert await stored(db_session, store, "32015R0757") == b"<html>v2</html>"


async def test_still_rendering_doc_fails_without_destroying_its_stored_bytes(
    db_session, store, corpus_client
):
    """The regression: a 202 used to be stored as an empty file, wiping the last good copy."""
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    _, first = await fetch(db_session, client, ["mrv"], store)
    keys = {document.celex: document.key for document in first}

    rendering = mrv_docs({"32015R0757": httpx.Response(202, content=b"")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, rendering)
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert "32015R0757" in report.failed
    assert not report.ok
    assert store.get(keys["32015R0757"]) == b"<html>mrv</html>"
    assert report.unchanged == ["32023R2449"]


async def test_vanished_doc_reported_dropped(db_session, store, corpus_client):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], store)

    only_seed = httpx.Response(200, json=payload(binding("32015R0757", force="1")))
    client, _ = corpus_client({"mrv": only_seed}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert report.dropped == ["32023R2449"]
    assert "32023R2449" not in await get_baseline_docs(db_session, ["mrv"])


async def test_per_doc_failure_continues_and_is_recorded(db_session, store, corpus_client):
    docs = mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert report.new == ["32015R0757"]
    assert "32023R2449" in report.failed
    assert not report.ok
    assert "32023R2449" not in await get_baseline_docs(db_session, ["mrv"])


async def test_a_storage_failure_is_recorded_per_document_not_raised(
    db_session, store, corpus_client, monkeypatch
):
    """A bucket that will not take writes fails the run, one recorded document at a time."""

    def unwritable(key, content):
        raise StorageError("put", key)

    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    monkeypatch.setattr(store, "put", unwritable)
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert sorted(report.failed) == ["32015R0757", "32023R2449"]
    assert not report.ok
    assert await get_baseline_docs(db_session, ["mrv"]) == {}


async def test_any_ingestion_error_is_recorded_per_document(
    db_session, store, corpus_client, monkeypatch
):
    """The per-document loop catches the whole IngestionError family, not just resolution."""
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())

    def unparseable(*args, **kwargs):
        raise ParseError("unrecognised EUR-Lex dialect")

    monkeypatch.setattr(stage, "_fetch_document", unparseable)
    report, _ = await fetch(db_session, client, ["mrv"], store)

    assert sorted(report.failed) == ["32015R0757", "32023R2449"]
    assert set(report.failed.values()) == {"ParseError: unrecognised EUR-Lex dialect"}
    assert not report.ok


async def test_duplicate_celex_across_topics_ingested_once(db_session, store, corpus_client):
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
    report, _ = await fetch(db_session, client, ["fueleu", "mrv"], store)

    assert report.ok
    rows = await get_baseline_docs(db_session, ["fueleu", "mrv"])
    assert rows["32015R0757"].topic == "fueleu"


async def test_paces_between_documents(db_session, store, corpus_client, paces):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await fetch(db_session, client, ["mrv"], store)
    assert paces == [PACE_SECONDS]


async def test_reuse_records_the_stored_corpus_against_the_new_run(
    db_session, store, corpus_client
):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    _, first = await fetch(db_session, client, ["mrv"], store)

    run = await create_ingest_run(db_session)
    documents, report = await reuse_documents(db_session, topics=["mrv"], run=run)

    assert sorted(report.unchanged) == ["32015R0757", "32023R2449"]
    assert sorted(report.discovered) == ["32015R0757", "32023R2449"]
    assert report.ok
    assert {document.ingest_run_id for document in documents} == {run.id}
    assert {document.key for document in documents} == {document.key for document in first}


async def test_reused_rows_still_point_at_the_stored_bytes(db_session, store, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await fetch(db_session, client, ["mrv"], store)

    run = await create_ingest_run(db_session)
    documents, _ = await reuse_documents(db_session, topics=["mrv"], run=run)

    by_celex = {document.celex: document for document in documents}
    assert store.get(by_celex["32015R0757"].key) == b"<html>mrv</html>"


async def test_reuse_refuses_when_nothing_has_been_fetched(db_session):
    run = await create_ingest_run(db_session)
    with pytest.raises(EmptyCorpusError, match="mrv"):
        await reuse_documents(db_session, topics=["mrv"], run=run)
