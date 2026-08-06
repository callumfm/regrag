"""The ingest orchestrator end to end: run lifecycle, corpus versioning, chunk persistence."""

import re
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.chunker import chunk_document
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.enums import IngestRunStatus, SectionKind
from app.ingestion.exceptions import DiscoveryError
from app.ingestion.fetch import stage
from app.ingestion.models import RunReport
from app.ingestion.parse.eurlex_html import parse_eurlex_html
from app.ingestion.pipeline import ingest
from app.ingestion.schemas import IngestRun
from tests.conftest import binding, payload

pytestmark = pytest.mark.anyio

MRV_SPARQL = httpx.Response(
    200, json=payload(binding("32015R0757", force="1"), binding("32023R2449", force="1"))
)

FUELEU_HTML = (Path(__file__).parent / "parse" / "fixtures" / "32023R1805.html").read_text()


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-ref responses overridable."""
    return {
        "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
        "32023R2449": httpx.Response(200, content=FUELEU_HTML.encode()),
    } | (overrides or {})


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


async def chunk_rows(session: AsyncSession, ref: str | None = None) -> list[DocumentChunk]:
    stmt = select(DocumentChunk).order_by(DocumentChunk.id)
    if ref is not None:
        stmt = stmt.where(DocumentChunk.ref == ref)
    return list(await session.scalars(stmt))


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


async def ingest_fueleu(db_session, tmp_path, corpus_client) -> RunReport:
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
