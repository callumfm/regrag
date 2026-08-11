"""The ingest orchestrator end to end: the run report, run lifecycle, corpus versioning."""

import re

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import ProgrammingError

from app.core.storage import StorageError
from app.ingestion import pipeline
from app.ingestion.celex import consolidated_stem
from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.enums import IngestRunStatus, SectionKind
from app.ingestion.exceptions import CorpusShrankError, MalformedDiscoveryError
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.service import get_previous_docs
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParseRunResult
from app.ingestion.pipeline import celexes_to_keep, ingest
from app.ingestion.result import IngestRunResult
from app.ingestion.schemas import IngestRun
from app.ingestion.storage import document_key
from tests.conftest import (
    FUELEU_HTML,
    MRV_SPARQL,
    binding,
    chunk_rows,
    chunk_versions,
    discovered_document,
    payload,
)

pytestmark = pytest.mark.anyio


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-celex responses overridable."""
    return {
        "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
        "32023R2449": httpx.Response(200, content=FUELEU_HTML.encode()),
    } | (overrides or {})


async def test_celexes_to_keep_unions_this_run_with_the_topics_it_left_alone(
    db_session, make_document
):
    other = IngestRun(status=IngestRunStatus.SUCCESS)
    db_session.add(make_document(other, "32015R0757", topic="mrv"))
    await db_session.flush()

    celexes = await celexes_to_keep(
        db_session,
        discovered=[discovered_document("32023R1805", topic="fueleu")],
        result=IngestRunResult(run_id=1, parse=ParseRunResult(parsed=["32023R1805"])),
        topics=["fueleu"],
    )

    assert celexes == {"32023R1805", "32015R0757"}


async def test_a_partial_fetch_leaves_the_corpus_unknown(db_session):
    result = IngestRunResult(
        run_id=1,
        fetch=FetchRunResult(failed={"32015R0757": "404"}),
        parse=ParseRunResult(parsed=["32023R1805"]),
    )

    celexes = await celexes_to_keep(
        db_session,
        discovered=[discovered_document("32023R1805", topic="fueleu")],
        result=result,
        topics=["fueleu"],
    )
    assert celexes is None


async def test_a_document_that_would_not_parse_leaves_the_corpus_unknown(db_session):
    result = IngestRunResult(run_id=1, parse=ParseRunResult(failed={"32023R1805": "ParseError"}))

    celexes = await celexes_to_keep(
        db_session,
        discovered=[discovered_document("32023R1805", topic="fueleu")],
        result=result,
        topics=["fueleu"],
    )
    assert celexes is None


async def test_sparql_failure_aborts_and_marks_run_aborted(db_session, local_store, corpus_client):
    client, _ = corpus_client({"mrv": httpx.Response(500, text="down")}, {})
    with pytest.raises(httpx.HTTPStatusError):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.ABORTED
    assert run.completed_at is not None


async def test_completed_run_is_stamped_with_a_dated_corpus_version(
    db_session, local_store, corpus_client
):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.SUCCESS
    assert run.completed_at is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{7}", run.corpus_version)


async def test_unchanged_corpus_keeps_the_previous_corpus_version(
    db_session, local_store, corpus_client
):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    first = await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] is not None
    assert versions[0] == versions[1]


async def test_changed_document_produces_a_new_corpus_version(
    db_session, local_store, corpus_client
):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    first = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1", cons="02015R0757-20250101"),
            binding("32023R2449", force="1"),
        ),
    )
    client, _ = corpus_client(
        {"mrv": consolidated},
        mrv_docs({"02015R0757-20250101": httpx.Response(200, content=FUELEU_HTML.encode())}),
    )
    second = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] != versions[1]


async def test_failed_run_is_not_stamped_with_a_corpus_version(
    db_session, local_store, corpus_client
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED
    assert run.corpus_version is None


async def test_a_completed_run_records_what_each_stage_did(db_session, local_store, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.result == report.report()
    assert run.result["fetch"]["new"] == 2
    assert run.result["embed"]["embedded"] > 0


async def test_a_stage_failure_is_answerable_from_the_run_row(
    db_session, local_store, corpus_client
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = await db_session.get(IngestRun, report.run_id)
    assert "32023R2449" in run.result["fetch"]["failed"]
    assert run.result["fetch"]["new"] == 1


async def test_a_run_aborted_before_any_stage_ran_records_every_stage_as_zero(
    db_session, local_store, corpus_client
):
    """Nothing ran, so every count is zero; the ABORTED status is what says the run stopped."""
    client, _ = corpus_client({"mrv": httpx.Response(500, text="down")}, {})
    with pytest.raises(httpx.HTTPStatusError):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.result == {
        "discover": {"dropped": 0, "failed": {}},
        "fetch": {"new": 0, "changed": 0, "unchanged": 0, "failed": {}},
        "parse": {"parsed": 0, "failed": {}},
        "chunk": {"added": 0, "removed": 0, "unchanged": 0, "failed": {}},
        "embed": {"embedded": 0, "unchanged": 0, "failed": {}},
    }


async def test_malformed_sparql_payload_raises_discovery_error(
    db_session, local_store, corpus_client
):
    client, _ = corpus_client({"mrv": httpx.Response(200, json={"unexpected": True})}, {})
    with pytest.raises(MalformedDiscoveryError, match="malformed"):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)


async def test_a_failure_after_fetch_still_marks_the_run_aborted(
    db_session, local_store, corpus_client, monkeypatch
):
    async def explode(*args, **kwargs):
        raise RuntimeError("chunking blew up")

    monkeypatch.setattr("app.ingestion.pipeline.chunk_and_store_document", explode)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(RuntimeError):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.ABORTED
    assert run.completed_at is not None


async def test_a_failure_closing_the_run_out_still_marks_it_aborted(
    db_session, local_store, corpus_client, monkeypatch
):
    """The closing commit is where an integrity error or a Ctrl-C lands, so it must be covered."""
    real = pipeline.complete_ingest_run
    calls = []

    async def explode_once(session, run, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("commit blew up")
        return await real(session, run, **kwargs)

    monkeypatch.setattr(pipeline, "complete_ingest_run", explode_once)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(RuntimeError, match="commit blew up"):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.ABORTED
    assert run.completed_at is not None


async def test_an_aborted_run_reports_exactly_the_documents_it_committed(
    db_session, local_store, corpus_client, monkeypatch
):
    """The row must not claim a document the abort rolled back: counts land after the commit."""
    real = pipeline.chunk_and_store_document
    calls: list[int] = []

    async def die_on_the_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("chunking blew up")
        return await real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "chunk_and_store_document", die_on_the_second)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(RuntimeError):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = (await db_session.scalars(select(IngestRun))).one()
    stored = (await db_session.scalars(select(RawDocument))).all()
    assert run.result["fetch"]["new"] == len(stored) == 1
    assert run.result["parse"]["parsed"] == 1
    assert run.status is IngestRunStatus.ABORTED


async def test_run_persists_chunks_for_every_document(db_session, local_store, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    rows = await chunk_rows(db_session)
    assert report.ok
    assert report.chunk.added == len(rows) > 0
    assert {row.celex for row in rows} == {"32015R0757", "32023R2449"}
    assert {row.ingest_run_id for row in rows} == {report.run_id}
    assert await chunk_versions(db_session) == {report.corpus_version}


async def test_second_identical_run_adds_and_removes_nothing(
    db_session, local_store, corpus_client
):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    before = {row.id for row in await chunk_rows(db_session)}

    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert (second.chunk.added, second.chunk.removed) == (0, 0)
    assert second.chunk.unchanged == len(before)
    assert {row.id for row in await chunk_rows(db_session)} == before


ONLY_SEED_SPARQL = httpx.Response(200, json=payload(binding("32015R0757", force="1")))


def new_version(celex: str) -> str:
    """A consolidated version the last run never saw, so the act has to be downloaded again."""
    return f"{consolidated_stem(celex)}20250101"


async def test_dropped_document_loses_its_chunks(db_session, local_store, corpus_client):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    assert await chunk_rows(db_session, "32023R2449")

    client, _ = corpus_client({"mrv": ONLY_SEED_SPARQL}, docs)
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert report.discover.dropped == ["32023R2449"]
    assert await chunk_rows(db_session, "32023R2449") == []
    assert await chunk_rows(db_session, "32015R0757")


async def test_dropped_document_loses_its_chunks_after_an_intervening_failed_fetch(
    db_session, local_store, corpus_client
):
    """A failed fetch drops the doc from what the next run treats as previous; chunks must still go.

    The failure has to land on a version the run actually downloads: an unchanged act is
    never requested, so it has no way to fail.
    """
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    assert await chunk_rows(db_session, "32023R2449")

    consolidated = httpx.Response(
        200,
        json=payload(
            binding("32015R0757", force="1"),
            binding("32023R2449", force="1", cons=new_version("32023R2449")),
        ),
    )
    client, _ = corpus_client(
        {"mrv": consolidated},
        mrv_docs({new_version("32023R2449"): httpx.Response(400, text="bad")}),
    )
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    client, _ = corpus_client({"mrv": ONLY_SEED_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert report.discover.dropped == []
    assert await chunk_rows(db_session, "32023R2449") == []
    assert await chunk_rows(db_session, "32015R0757")


WIDE_CELEXES = ["32015R0757", "32023R2449", "32026R0001", "32026R0002", "32026R0003"]
WIDE_SPARQL = httpx.Response(
    200, json=payload(*(binding(celex, force="1") for celex in WIDE_CELEXES))
)


def wide_docs() -> dict[str, httpx.Response]:
    """A five-document mrv corpus, enough that losing most of it is implausible."""
    return {celex: httpx.Response(200, content=FUELEU_HTML.encode()) for celex in WIDE_CELEXES}


async def test_discovery_losing_most_of_the_corpus_aborts_and_deletes_nothing(
    db_session, local_store, corpus_client
):
    """A truncated result set reads exactly like a mass repeal, so refuse to act on it."""
    client, _ = corpus_client({"mrv": WIDE_SPARQL}, wide_docs())
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    before = {row.id for row in await chunk_rows(db_session)}
    assert before

    client, _ = corpus_client({"mrv": ONLY_SEED_SPARQL}, wide_docs())
    with pytest.raises(CorpusShrankError, match="lost 4 of 5"):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert {row.id for row in await chunk_rows(db_session)} == before


async def test_a_plausible_repeal_still_prunes(db_session, local_store, corpus_client):
    """One document out of five is an ordinary repeal, not a truncated response."""
    client, _ = corpus_client({"mrv": WIDE_SPARQL}, wide_docs())
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    kept = [binding(celex, force="1") for celex in WIDE_CELEXES if celex != "32026R0003"]
    client, _ = corpus_client({"mrv": httpx.Response(200, json=payload(*kept))}, wide_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert report.discover.dropped == ["32026R0003"]
    assert await chunk_rows(db_session, "32026R0003") == []


async def test_an_incomplete_run_prunes_nothing(db_session, local_store, corpus_client):
    """A run that could not fetch its whole corpus has no business declaring anything obsolete."""
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    assert await chunk_rows(db_session, "32023R2449")

    seed_consolidated = httpx.Response(
        200, json=payload(binding("32015R0757", force="1", cons=new_version("32015R0757")))
    )
    client, _ = corpus_client(
        {"mrv": seed_consolidated},
        mrv_docs({new_version("32015R0757"): httpx.Response(400, text="bad")}),
    )
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert report.discover.dropped == ["32023R2449"]
    assert not report.ok
    assert report.chunk.removed == 0
    assert await chunk_rows(db_session, "32023R2449")


FUELEU_PLUS_SHARED = httpx.Response(
    200, json=payload(binding("32023R1805", force="1"), binding("32015R0757", force="1"))
)


async def test_a_celex_another_topic_still_holds_survives_being_dropped(
    db_session, local_store, corpus_client
):
    """32015R0757 is tagged fueleu because fueleu saw it first; mrv still wanting it must win."""
    shared_docs = {
        "32023R1805": httpx.Response(200, content=FUELEU_HTML.encode()),
        "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
    }
    client, _ = corpus_client({"fueleu": FUELEU_PLUS_SHARED}, shared_docs)
    await ingest(db_session, client=client, topics=["fueleu"], store=local_store)
    assert {row.topic for row in await chunk_rows(db_session, "32015R0757")} == {"fueleu"}

    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    client, _ = corpus_client({"fueleu": FUELEU_SPARQL}, shared_docs)
    report = await ingest(db_session, client=client, topics=["fueleu"], store=local_store)

    assert report.discover.dropped == ["32015R0757"]
    assert await chunk_rows(db_session, "32015R0757")


async def test_unparseable_document_is_recorded_and_others_persist(
    db_session, local_store, corpus_client
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        mrv_docs({"32023R2449": httpx.Response(200, content=b"<html>not eur-lex</html>")}),
    )
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert "32023R2449" in report.parse.failed
    assert not report.ok
    assert await chunk_rows(db_session, "32015R0757")
    assert await chunk_rows(db_session, "32023R2449") == []
    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED


async def test_a_freshly_downloaded_document_is_parsed_without_reading_it_back(
    db_session, local_store, corpus_client, monkeypatch
):
    """The download already holds the bytes, so parse must not pay a second storage round trip."""
    reads: list[str] = []
    monkeypatch.setattr(local_store, "get", lambda key: reads.append(key))
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert reads == []
    assert report.ok


async def test_a_source_document_lost_from_the_store_is_downloaded_again(
    db_session, local_store, corpus_client
):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    for path in local_store.root.rglob("*.html"):
        path.unlink()

    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    rows = await get_previous_docs(db_session, ["mrv"])
    row = rows["32015R0757"]
    assert local_store.exists(document_key(row.celex, row.resolved_celex, row.sha256))
    assert report.ok


async def test_a_store_write_failure_is_recorded_not_raised(
    db_session, local_store, corpus_client, monkeypatch
):
    def full_disk(key, content):
        raise StorageError("put", key, OSError(28, "No space left on device"))

    monkeypatch.setattr(local_store, "put", full_disk)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert sorted(report.fetch.failed) == ["32015R0757", "32023R2449"]
    assert all("StorageError" in reason for reason in report.fetch.failed.values())
    assert not report.ok


async def test_an_interrupt_still_marks_the_run_aborted(
    db_session, local_store, corpus_client, monkeypatch
):
    async def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("app.ingestion.pipeline.chunk_and_store_document", interrupt)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(KeyboardInterrupt):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.ABORTED
    assert run.completed_at is not None


async def test_a_database_failure_does_not_mask_itself(
    db_session, local_store, corpus_client, monkeypatch
):
    """The failure handler must roll back first, or its own commit raises over the real cause."""

    async def explode(*args, **kwargs):
        await db_session.execute(text("SELECT * FROM no_such_table"))

    monkeypatch.setattr("app.ingestion.pipeline.chunk_and_store_document", explode)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(ProgrammingError):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.ABORTED


async def test_partial_fetch_leaves_its_chunks_unstamped(db_session, local_store, corpus_client):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert report.corpus_version is None
    assert await chunk_versions(db_session, "32015R0757") == {None}


async def test_chunks_stay_attributed_to_the_run_that_first_stored_them(
    db_session, local_store, corpus_client
):
    """A later whole-corpus run leaves matched chunks pointing at the run they came from.

    Those chunks resolve to no corpus version, because the run that stored them failed.
    """
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    partial = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert report.corpus_version is not None
    rows = await chunk_rows(db_session, "32015R0757")
    assert {row.ingest_run_id for row in rows} == {partial.run_id}
    assert await chunk_versions(db_session, "32015R0757") == {None}


async def test_failed_fetch_still_chunks_what_was_downloaded(
    db_session, local_store, corpus_client
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert "32023R2449" in report.fetch.failed
    assert not report.ok
    assert report.corpus_version is None
    assert await chunk_rows(db_session, "32015R0757")


FUELEU_SPARQL = httpx.Response(200, json=payload(binding("32023R1805", force="1")))


async def ingest_fueleu(db_session, local_store, corpus_client) -> IngestRunResult:
    """Run the real pipeline over the saved FuelEU fixture, network stubbed."""
    client, _ = corpus_client(
        {"fueleu": FUELEU_SPARQL},
        {"32023R1805": httpx.Response(200, content=FUELEU_HTML.encode())},
    )
    return await ingest(db_session, client=client, topics=["fueleu"], store=local_store)


async def test_fueleu_chunks_have_correct_article_boundaries(
    db_session, local_store, corpus_client
):
    report = await ingest_fueleu(db_session, local_store, corpus_client)
    assert report.ok

    rows = await chunk_rows(db_session, "32023R1805")
    boundaries = [(row.article, row.paragraph) for row in rows if row.article]
    assert boundaries == [("4", str(n)) for n in range(1, 5)] + [
        ("5", str(n)) for n in range(1, 11)
    ]


async def test_fueleu_chunks_carry_their_citations(db_session, local_store, corpus_client):
    await ingest_fueleu(db_session, local_store, corpus_client)

    rows = await chunk_rows(db_session, "32023R1805")
    citations = {row.citation for row in rows}
    assert "Article 4(1)" in citations
    assert "Annex II" in citations


async def test_fueleu_annex_table_is_persisted_as_its_own_chunk(
    db_session, local_store, corpus_client
):
    await ingest_fueleu(db_session, local_store, corpus_client)

    rows = await chunk_rows(db_session, "32023R1805")
    tables = [row for row in rows if row.kind is SectionKind.TABLE]
    assert len(tables) == 1
    assert tables[0].citation == "Annex II"


async def test_fueleu_chunks_are_stamped_and_topic_tagged(db_session, local_store, corpus_client):
    report = await ingest_fueleu(db_session, local_store, corpus_client)

    rows = await chunk_rows(db_session, "32023R1805")
    assert {row.topic for row in rows} == {"fueleu"}
    assert await chunk_versions(db_session, "32023R1805") == {report.corpus_version}
    assert report.corpus_version is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{7}", report.corpus_version)


async def test_single_topic_run_leaves_another_topics_chunks_alone(
    db_session, local_store, corpus_client
):
    await ingest_fueleu(db_session, local_store, corpus_client)
    before = {row.id for row in await chunk_rows(db_session, "32023R1805")}
    assert before

    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert {row.id for row in await chunk_rows(db_session, "32023R1805")} == before


async def test_fueleu_chunk_count_matches_the_chunker(db_session, local_store, corpus_client):
    """Persisted order matches the chunker for a fresh corpus only: ids are first-insert order,
    so after any partial change replacement rows sort above untouched ones."""
    await ingest_fueleu(db_session, local_store, corpus_client)

    expected = chunk_document(parse_eurlex_html(FUELEU_HTML, "32023R1805", "fueleu"))
    rows = await chunk_rows(db_session, "32023R1805")
    assert len(rows) == len(expected)
    assert [row.text for row in rows] == [c.text for c in expected]


async def test_a_run_embeds_every_chunk_it_stored(db_session, local_store, corpus_client):
    report = await ingest_fueleu(db_session, local_store, corpus_client)

    rows = await chunk_rows(db_session, "32023R1805")
    assert report.ok
    assert report.embed.embedded == len(rows) > 0
    assert all(row.embedding is not None for row in rows)


async def test_a_second_run_embeds_nothing_and_reports_the_rest_unchanged(
    db_session, local_store, corpus_client
):
    await ingest_fueleu(db_session, local_store, corpus_client)
    stored = len(await chunk_rows(db_session, "32023R1805"))

    second = await ingest_fueleu(db_session, local_store, corpus_client)

    assert second.ok
    assert second.embed.failed == {}
    assert (second.embed.embedded, second.embed.unchanged) == (0, stored)


async def test_a_document_that_fails_does_not_stop_the_documents_after_it(
    db_session, local_store, corpus_client
):
    """The loop is per document across three stages: one bad document skips only itself."""
    docs = mrv_docs({"32015R0757": httpx.Response(400, text="bad")})
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)

    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert list(report.fetch.failed) == ["32015R0757"]
    assert report.fetch.new == ["32023R2449"]
    assert report.parse.parsed == ["32023R2449"]
    assert report.chunk.added > 0
    assert not report.ok


async def test_a_run_that_died_mid_loop_does_not_stand_for_its_topics_corpus(
    db_session, local_store, corpus_client, monkeypatch
):
    """An aborted run holds a prefix, not a corpus: another topic must not prune the rest."""
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client=client, topics=["mrv"], store=local_store)
    assert await chunk_rows(db_session, "32023R2449")

    real = pipeline.chunk_and_store_document
    calls: list[int] = []

    async def die_on_the_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return await real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "chunk_and_store_document", die_on_the_second)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    with pytest.raises(KeyboardInterrupt):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    monkeypatch.setattr(pipeline, "chunk_and_store_document", real)
    client, _ = corpus_client(
        {"fueleu": FUELEU_SPARQL},
        {"32023R1805": httpx.Response(200, content=FUELEU_HTML.encode())},
    )
    await ingest(db_session, client=client, topics=["fueleu"], store=local_store)

    assert await chunk_rows(db_session, "32023R2449")


async def test_a_run_that_dies_in_embed_keeps_its_documents_and_chunks(
    db_session, local_store, corpus_client, monkeypatch
):
    """A late failure must not discard the fetch and chunk work the run already committed."""

    real_embed_chunks = pipeline.embed_chunks

    async def provider_gone(session):
        raise RuntimeError("provider gone")

    monkeypatch.setattr(pipeline, "embed_chunks", provider_gone)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())

    with pytest.raises(RuntimeError):
        await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert await chunk_versions(db_session) != set()
    assert set(await get_previous_docs(db_session, ["mrv"])) == {"32015R0757", "32023R2449"}

    monkeypatch.setattr(pipeline, "embed_chunks", real_embed_chunks)
    stored = len(await chunk_rows(db_session))
    client, calls = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    second = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert calls == []
    assert second.embed.embedded == stored
    assert second.chunk.added == 0
    assert second.ok


async def test_a_run_where_every_document_fails_still_reports_the_later_stages(
    db_session, local_store, corpus_client, monkeypatch
):
    """Parse and chunk had nothing to do; the row reports that as zeroes, not as a failure."""

    def full_disk(key, content):
        raise StorageError("put", key, OSError(28, "No space left on device"))

    monkeypatch.setattr(local_store, "put", full_disk)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())

    report = await ingest(db_session, client=client, topics=["mrv"], store=local_store)

    assert report.report()["parse"] == {"parsed": 0, "failed": {}}
    assert report.report()["chunk"] == {"added": 0, "removed": 0, "unchanged": 0, "failed": {}}
    assert not report.ok
