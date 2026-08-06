"""The ingest orchestrator end to end: the run report, run lifecycle, corpus versioning."""

import re
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import ProgrammingError

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.enums import IngestRunStatus, SectionKind
from app.ingestion.exceptions import DiscoveryError
from app.ingestion.fetch import stage
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.parse.eurlex_html import parse_eurlex_html
from app.ingestion.parse.models import ParseRunResult
from app.ingestion.pipeline import IngestRunResult, ingest
from app.ingestion.schemas import IngestRun
from tests.conftest import MRV_SPARQL, binding, chunk_rows, payload

pytestmark = pytest.mark.anyio

FUELEU_HTML = (Path(__file__).parent / "parse" / "fixtures" / "32023R1805.html").read_text()


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-ref responses overridable."""
    return {
        "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
        "32023R2449": httpx.Response(200, content=FUELEU_HTML.encode()),
    } | (overrides or {})


def test_stages_finds_every_result_and_nothing_else() -> None:
    assert list(IngestRunResult(run_id=1).stages) == ["fetch", "parse", "chunk"]


def test_a_run_is_ok_when_no_stage_failed() -> None:
    assert IngestRunResult(run_id=1).ok
    assert IngestRunResult(run_id=1).status is IngestRunStatus.COMPLETED


def test_a_failure_in_any_stage_fails_the_run() -> None:
    assert not IngestRunResult(run_id=1, fetch=FetchRunResult(failed={"a": "404"})).ok
    assert not IngestRunResult(run_id=1, parse=ParseRunResult(failed={"a": "ParseError"})).ok
    assert IngestRunResult(run_id=1, chunk=ChunkRunResult(failed={"a": "boom"})).status is (
        IngestRunStatus.FAILED
    )


def test_summary_reports_every_stage_on_its_own_line() -> None:
    result = IngestRunResult(
        run_id=7,
        corpus_version="2026-08-05-abc1234",
        fetch=FetchRunResult(new=["a"], unchanged=["b"]),
        parse=ParseRunResult(parsed=["a", "b"]),
        chunk=ChunkRunResult(added=12, unchanged=30),
    )
    assert result.summary().splitlines() == [
        "run 7 (2026-08-05-abc1234)",
        "  [fetch] 1 new, 0 changed, 1 unchanged, 0 dropped, 0 failed",
        "  [parse] 2 parsed, 0 failed",
        "  [chunk] 12 added, 0 removed, 30 unchanged, 0 failed",
        "  fetch new: a",
    ]


def test_summary_says_so_when_no_version_was_stamped() -> None:
    assert IngestRunResult(run_id=7).summary().startswith("run 7 (not stamped)")


def test_summary_lists_each_stage_s_failures() -> None:
    result = IngestRunResult(
        run_id=7,
        fetch=FetchRunResult(failed={"a": "HTTPError: 404"}),
        parse=ParseRunResult(failed={"b": "ParseError: no body"}),
    )
    assert "  fetch failed: a (HTTPError: 404)" in result.summary()
    assert "  parse failed: b (ParseError: no body)" in result.summary()


async def test_sparql_failure_aborts_and_marks_run_failed(db_session, tmp_path, corpus_client):
    client, _ = corpus_client({"mrv": httpx.Response(500, text="down")}, {})
    with pytest.raises(httpx.HTTPStatusError):
        await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None


async def test_completed_run_is_stamped_with_a_dated_corpus_version(
    db_session, tmp_path, corpus_client
):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.COMPLETED
    assert run.completed_at is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{7}", run.corpus_version)


async def test_unchanged_corpus_keeps_the_previous_corpus_version(
    db_session, tmp_path, corpus_client
):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    first = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] is not None
    assert versions[0] == versions[1]


async def test_changed_document_produces_a_new_corpus_version(db_session, tmp_path, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    first = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

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
    second = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] != versions[1]


async def test_failed_run_is_not_stamped_with_a_corpus_version(db_session, tmp_path, corpus_client):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED
    assert run.corpus_version is None


async def test_malformed_sparql_payload_raises_discovery_error(db_session, tmp_path, corpus_client):
    client, _ = corpus_client({"mrv": httpx.Response(200, json={"unexpected": True})}, {})
    with pytest.raises(DiscoveryError, match="malformed"):
        await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)


async def test_a_failure_after_fetch_still_marks_the_run_failed(
    db_session, tmp_path, corpus_client, monkeypatch
):
    async def explode(*args, **kwargs):
        raise RuntimeError("chunking blew up")

    monkeypatch.setattr("app.ingestion.pipeline.chunk_documents", explode)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(RuntimeError):
        await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None


async def test_run_persists_chunks_for_every_document(db_session, tmp_path, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    rows = await chunk_rows(db_session)
    assert report.ok
    assert report.chunk.added == len(rows) > 0
    assert {row.ref for row in rows} == {"32015R0757", "32023R2449"}
    assert {row.corpus_version for row in rows} == {report.corpus_version}


async def test_second_identical_run_adds_and_removes_nothing(db_session, tmp_path, corpus_client):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)
    before = {row.id for row in await chunk_rows(db_session)}

    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert (second.chunk.added, second.chunk.removed) == (0, 0)
    assert second.chunk.unchanged == len(before)
    assert {row.id for row in await chunk_rows(db_session)} == before


ONLY_SEED_SPARQL = httpx.Response(200, json=payload(binding("32015R0757", force="1")))


async def test_dropped_document_loses_its_chunks(db_session, tmp_path, corpus_client):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)
    assert await chunk_rows(db_session, "32023R2449")

    client, _ = corpus_client({"mrv": ONLY_SEED_SPARQL}, docs)
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert report.fetch.dropped == ["32023R2449"]
    assert await chunk_rows(db_session, "32023R2449") == []
    assert await chunk_rows(db_session, "32015R0757")


async def test_dropped_document_loses_its_chunks_after_an_intervening_failed_fetch(
    db_session, tmp_path, corpus_client
):
    """A failed fetch drops the doc from the next run's baseline; chunks must still go."""
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)
    assert await chunk_rows(db_session, "32023R2449")

    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    client, _ = corpus_client({"mrv": ONLY_SEED_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert report.fetch.dropped == []
    assert await chunk_rows(db_session, "32023R2449") == []
    assert await chunk_rows(db_session, "32015R0757")


async def test_unparseable_document_is_recorded_and_others_persist(
    db_session, tmp_path, corpus_client
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        mrv_docs({"32023R2449": httpx.Response(200, content=b"<html>not eur-lex</html>")}),
    )
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert "32023R2449" in report.parse.failed
    assert not report.ok
    assert await chunk_rows(db_session, "32015R0757")
    assert await chunk_rows(db_session, "32023R2449") == []
    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED


async def test_missing_source_file_is_recorded_not_raised(
    db_session, tmp_path, corpus_client, monkeypatch
):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    monkeypatch.setattr(stage, "_store", lambda data_dir, ref, content: ("a" * 64, len(content)))
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert sorted(report.parse.failed) == ["32015R0757", "32023R2449"]
    assert all("FileNotFoundError" in reason for reason in report.parse.failed.values())
    assert not report.ok


async def test_a_source_file_lost_from_disk_is_downloaded_again(
    db_session, tmp_path, corpus_client
):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)
    for path in tmp_path.glob("*.html"):
        path.unlink()

    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert (tmp_path / "32015R0757.html").exists()
    assert report.ok


async def test_a_disk_error_is_recorded_not_raised(
    db_session, tmp_path, corpus_client, monkeypatch
):
    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(stage, "_store", full_disk)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert sorted(report.fetch.failed) == ["32015R0757", "32023R2449"]
    assert all("OSError" in reason for reason in report.fetch.failed.values())
    assert not report.ok


async def test_an_interrupt_still_marks_the_run_failed(
    db_session, tmp_path, corpus_client, monkeypatch
):
    async def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("app.ingestion.pipeline.chunk_documents", interrupt)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(KeyboardInterrupt):
        await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None


async def test_a_database_failure_does_not_mask_itself(
    db_session, tmp_path, corpus_client, monkeypatch
):
    """The failure handler must roll back first, or its own commit raises over the real cause."""

    async def explode(*args, **kwargs):
        await db_session.execute(text("SELECT * FROM no_such_table"))

    monkeypatch.setattr("app.ingestion.pipeline.chunk_documents", explode)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    with pytest.raises(ProgrammingError):
        await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.FAILED


async def test_partial_fetch_leaves_its_chunks_unstamped(db_session, tmp_path, corpus_client):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert report.corpus_version is None
    assert {row.corpus_version for row in await chunk_rows(db_session, "32015R0757")} == {None}


async def test_a_whole_corpus_run_backfills_chunks_left_unstamped(
    db_session, tmp_path, corpus_client
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert report.corpus_version is not None
    assert {row.corpus_version for row in await chunk_rows(db_session, "32015R0757")} == {
        report.corpus_version
    }


async def test_failed_fetch_still_chunks_what_was_downloaded(db_session, tmp_path, corpus_client):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert "32023R2449" in report.fetch.failed
    assert not report.ok
    assert report.corpus_version is None
    assert await chunk_rows(db_session, "32015R0757")


FUELEU_SPARQL = httpx.Response(200, json=payload(binding("32023R1805", force="1")))


async def ingest_fueleu(db_session, tmp_path, corpus_client) -> IngestRunResult:
    """Run the real pipeline over the saved FuelEU fixture, network stubbed."""
    client, _ = corpus_client(
        {"fueleu": FUELEU_SPARQL},
        {"32023R1805": httpx.Response(200, content=FUELEU_HTML.encode())},
    )
    return await ingest(db_session, client=client, topics=["fueleu"], data_dir=tmp_path)


async def test_fueleu_chunks_have_correct_article_boundaries(db_session, tmp_path, corpus_client):
    report = await ingest_fueleu(db_session, tmp_path, corpus_client)
    assert report.ok

    rows = await chunk_rows(db_session, "32023R1805")
    boundaries = [(row.article, row.paragraph) for row in rows if row.article]
    assert boundaries == [("4", str(n)) for n in range(1, 5)] + [
        ("5", str(n)) for n in range(1, 11)
    ]


async def test_fueleu_chunks_carry_their_citations(db_session, tmp_path, corpus_client):
    await ingest_fueleu(db_session, tmp_path, corpus_client)

    rows = await chunk_rows(db_session, "32023R1805")
    citations = {row.citation for row in rows}
    assert "Article 4(1)" in citations
    assert "Annex II" in citations


async def test_fueleu_annex_table_is_persisted_as_its_own_chunk(
    db_session, tmp_path, corpus_client
):
    await ingest_fueleu(db_session, tmp_path, corpus_client)

    rows = await chunk_rows(db_session, "32023R1805")
    tables = [row for row in rows if row.kind is SectionKind.TABLE]
    assert len(tables) == 1
    assert tables[0].citation == "Annex II"


async def test_fueleu_chunks_are_stamped_and_topic_tagged(db_session, tmp_path, corpus_client):
    report = await ingest_fueleu(db_session, tmp_path, corpus_client)

    rows = await chunk_rows(db_session, "32023R1805")
    assert {row.topic for row in rows} == {"fueleu"}
    assert {row.corpus_version for row in rows} == {report.corpus_version}
    assert report.corpus_version is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{7}", report.corpus_version)


async def test_single_topic_run_leaves_another_topics_chunks_alone(
    db_session, tmp_path, corpus_client
):
    await ingest_fueleu(db_session, tmp_path, corpus_client)
    before = {row.id for row in await chunk_rows(db_session, "32023R1805")}
    assert before

    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    await ingest(db_session, client=client, topics=["mrv"], data_dir=tmp_path)

    assert {row.id for row in await chunk_rows(db_session, "32023R1805")} == before


async def test_fueleu_chunk_count_matches_the_chunker(db_session, tmp_path, corpus_client):
    """Persisted order matches the chunker for a fresh corpus only: ids are first-insert order,
    so after any partial change replacement rows sort above untouched ones."""
    await ingest_fueleu(db_session, tmp_path, corpus_client)

    expected = chunk_document(parse_eurlex_html(FUELEU_HTML, "32023R1805", "fueleu"))
    rows = await chunk_rows(db_session, "32023R1805")
    assert len(rows) == len(expected)
    assert [row.text for row in rows] == [c.text for c in expected]
