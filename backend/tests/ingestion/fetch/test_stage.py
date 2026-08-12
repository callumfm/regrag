"""Reusing or downloading a document's version, and the fetch stage in isolation."""

import hashlib

import httpx
import pytest

from app.ingestion.discover.stage import discover_corpus
from app.ingestion.enums import DocChange, IngestRunStatus
from app.ingestion.exceptions import DocumentFailed, ParseError
from app.ingestion.fetch import stage
from app.ingestion.fetch.models import FetchedDocument
from app.ingestion.fetch.service import get_previous_docs
from app.ingestion.fetch.stage import _reuse_stored_version, fetch_document
from app.ingestion.schemas import IngestRun
from app.ingestion.service import complete_ingest_run, create_ingest_run
from app.ingestion.storage import document_key, read_document
from tests.conftest import MRV_SPARQL, binding, discovered_document, payload

pytestmark = pytest.mark.anyio


def stored(store_document, celex="32023R1805"):
    """A previous run's row whose bytes are still in the store."""
    return store_document(IngestRun(status=IngestRunStatus.SUCCESS), celex=celex)


def html_of(fetched, celex, local_store) -> bytes:
    """The bytes the run left in the store for one celex, read back the way a later run would."""
    return read_document(local_store, {f.document.celex: f.document for f in fetched}[celex])


def test_stored_version_is_reused_when_discovery_still_points_at_it(local_store, store_document):
    prev = stored(store_document)
    reused = _reuse_stored_version(local_store, discovered_document("32023R1805"), prev)
    assert reused is not None
    resolution, bytes_, content = reused
    assert resolution.resolved_celex == prev.resolved_celex
    assert content == b"<html>act</html>"
    assert (bytes_.sha256, bytes_.size_bytes, bytes_.fetched_at) == (
        prev.sha256,
        prev.size_bytes,
        prev.fetched_at,
    )


def test_a_newly_discovered_consolidation_is_not_reused(local_store, store_document):
    """Discovery pointing somewhere new is exactly the case that has to hit the network."""
    prev = stored(store_document)
    newer = discovered_document("32023R1805", candidate="02023R1805-20250101")
    assert _reuse_stored_version(local_store, newer, prev) is None


def test_stored_version_no_longer_in_the_store_is_not_reused(local_store, store_document):
    prev = stored(store_document)
    (local_store.root / document_key(prev.celex, prev.resolved_celex, prev.sha256)).unlink()
    assert _reuse_stored_version(local_store, discovered_document("32023R1805"), prev) is None


def test_a_document_with_no_previous_run_has_nothing_to_reuse(local_store):
    assert _reuse_stored_version(local_store, discovered_document("32023R1805"), None) is None


def test_stored_bytes_that_do_not_match_the_row_are_not_reused(local_store, store_document):
    """A row and an object restored from different points in time: download it again."""
    prev = stored(store_document)
    key = document_key(prev.celex, prev.resolved_celex, prev.sha256)
    local_store.put(key, b"<html>a different version</html>")

    assert _reuse_stored_version(local_store, discovered_document("32023R1805"), prev) is None


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-celex responses overridable."""
    return {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
    } | (overrides or {})


Fetched = tuple[dict[str, DocChange], dict[str, str], list[FetchedDocument]]


async def fetch(db_session, client, topics, store) -> Fetched:
    """Drive the fetch stage alone, with the run, discovery and savepoint the pipeline supplies."""
    run = await create_ingest_run(db_session)
    previous = await get_previous_docs(db_session, topics)
    discovered, _ = await discover_corpus(client, topics=topics, previous_celexes=previous)
    fetched: list[FetchedDocument] = []
    changes: dict[str, DocChange] = {}
    failed: dict[str, str] = {}
    for document in discovered:
        try:
            async with db_session.begin_nested():
                item, change = await fetch_document(
                    db_session,
                    client=client,
                    discovered=document,
                    previous=previous.get(document.celex),
                    run=run,
                    store=store,
                )
        except DocumentFailed as failure:
            failed[failure.celex] = failure.reason
        else:
            fetched.append(item)
            changes[document.celex] = change
    await complete_ingest_run(db_session, run, status=IngestRunStatus.SUCCESS)
    return changes, failed, fetched


def celexes(changes: dict[str, DocChange], change: DocChange) -> list[str]:
    """The celexes the run put in one of fetch's buckets."""
    return sorted(celex for celex, value in changes.items() if value is change)


async def test_first_run_ingests_all_as_new(db_session, local_store, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    changes, failed, documents = await fetch(db_session, client, ["mrv"], local_store)

    assert celexes(changes, DocChange.NEW) == ["32015R0757", "32023R2449"]
    assert failed == {}
    assert html_of(documents, "32015R0757", local_store) == b"<html>mrv</html>"
    rows = await get_previous_docs(db_session, ["mrv"])
    assert rows["32023R2449"].celex == "32023R2449"
    assert rows["32023R2449"].resolved_celex == "32023R2449"


async def test_unchanged_run_makes_no_html_requests_and_carries_sha(
    db_session, local_store, corpus_client
):
    """Steady state: discovery points at the versions already stored, so nothing is downloaded."""
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], local_store)

    client, calls = corpus_client({"mrv": MRV_SPARQL}, docs)
    changes, _, _ = await fetch(db_session, client, ["mrv"], local_store)

    assert celexes(changes, DocChange.REUSED) == ["32015R0757", "32023R2449"]
    assert calls == []
    firsts = {r.celex: r.sha256 for r in (await get_previous_docs(db_session, ["mrv"])).values()}
    assert firsts["32015R0757"] == hashlib.sha256(b"<html>mrv</html>").hexdigest()


async def test_new_consolidation_is_updated_and_redownloaded(
    db_session, local_store, corpus_client
):
    """One request, not two: the download hands back the bytes it already pulled."""
    docs = mrv_docs({"32015R0757": httpx.Response(200, content=b"<html>v1</html>")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], local_store)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    docs = mrv_docs({"02015R0757-20250101": httpx.Response(200, content=b"<html>v2</html>")})
    client, calls = corpus_client({"mrv": consolidated}, docs)
    changes, _, documents = await fetch(db_session, client, ["mrv"], local_store)

    assert celexes(changes, DocChange.UPDATED) == ["32015R0757"]
    assert calls == ["02015R0757-20250101"]
    assert html_of(documents, "32015R0757", local_store) == b"<html>v2</html>"
    rows = await get_previous_docs(db_session, ["mrv"])
    assert rows["32015R0757"].resolved_celex == "02015R0757-20250101"


async def test_still_rendering_doc_fails_leaving_the_parsed_bytes_readable(
    db_session, local_store, corpus_client
):
    """The regression: a 202 used to be stored as an empty file, wiping the last good copy.

    A new consolidation is what forces the download; an unchanged act is never requested at all.
    """
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    _, _, first = await fetch(db_session, client, ["mrv"], local_store)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    rendering = mrv_docs({"02015R0757-20250101": httpx.Response(202, content=b"")})
    client, _ = corpus_client({"mrv": consolidated}, rendering)
    changes, failed, _ = await fetch(db_session, client, ["mrv"], local_store)

    assert "32015R0757" in failed
    assert html_of(first, "32015R0757", local_store) == b"<html>mrv</html>"
    assert celexes(changes, DocChange.REUSED) == ["32023R2449"]


async def test_a_vanished_doc_gets_no_row_from_this_run(db_session, local_store, corpus_client):
    """Discovery is what reports the drop; fetch's part is simply never recording it again."""
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await fetch(db_session, client, ["mrv"], local_store)

    only_seed = httpx.Response(200, json=payload(binding("32015R0757", force="1")))
    client, _ = corpus_client({"mrv": only_seed}, docs)
    changes, _, _ = await fetch(db_session, client, ["mrv"], local_store)

    assert celexes(changes, DocChange.REUSED) == ["32015R0757"]
    assert "32023R2449" not in await get_previous_docs(db_session, ["mrv"])


async def test_per_doc_failure_continues_and_is_recorded(db_session, local_store, corpus_client):
    docs = mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    changes, failed, _ = await fetch(db_session, client, ["mrv"], local_store)

    assert celexes(changes, DocChange.NEW) == ["32015R0757"]
    assert "32023R2449" in failed
    assert "32023R2449" not in await get_previous_docs(db_session, ["mrv"])


async def test_any_ingestion_error_is_recorded_per_document(
    db_session, local_store, corpus_client, monkeypatch
):
    """The per-document loop catches the whole IngestionError family, not just resolution."""
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())

    def unparseable(*args, **kwargs):
        raise ParseError("unrecognised EUR-Lex dialect")

    monkeypatch.setattr(stage, "_reuse_or_download", unparseable)
    _, failed, _ = await fetch(db_session, client, ["mrv"], local_store)

    assert sorted(failed) == ["32015R0757", "32023R2449"]
    assert set(failed.values()) == {"ParseError: unrecognised EUR-Lex dialect"}


async def test_a_row_that_will_not_flush_fails_only_its_own_document(
    db_session, local_store, corpus_client, monkeypatch
):
    """A database error on one row must skip that document, not poison the run's transaction."""
    real = stage._reuse_or_download

    async def unstorable(client, discovered, **kwargs):
        fetched, change = await real(client, discovered, **kwargs)
        if discovered.celex == "32015R0757":
            fetched.document.topic = None  # ty: ignore[invalid-assignment]
        return fetched, change

    monkeypatch.setattr(stage, "_reuse_or_download", unstorable)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    changes, failed, _ = await fetch(db_session, client, ["mrv"], local_store)

    assert list(failed) == ["32015R0757"]
    assert celexes(changes, DocChange.NEW) == ["32023R2449"]


async def test_duplicate_celex_across_topics_ingested_once(db_session, local_store, corpus_client):
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
    _, failed, _ = await fetch(db_session, client, ["fueleu", "mrv"], local_store)

    assert failed == {}
    rows = await get_previous_docs(db_session, ["fueleu", "mrv"])
    assert rows["32015R0757"].topic == "fueleu"
