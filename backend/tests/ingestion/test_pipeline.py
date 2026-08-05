"""The ingest orchestrator: run lifecycle and corpus versioning around the fetch stage."""

import re
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import DiscoveryError
from app.ingestion.fetch import corpus
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
        await ingest(db_session, client, ["mrv"], tmp_path)

    run = (await db_session.scalars(select(IngestRun))).one()
    assert run.status is IngestRunStatus.FAILED
    assert run.completed_at is not None


async def test_completed_run_is_stamped_with_a_dated_corpus_version(
    db_session, tmp_path, corpus_client
):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    report = await ingest(db_session, client, ["mrv"], tmp_path)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.COMPLETED
    assert run.completed_at is not None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{7}", run.corpus_version)


async def test_unchanged_corpus_keeps_the_previous_corpus_version(
    db_session, tmp_path, corpus_client
):
    docs = mrv_docs()
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    first = await ingest(db_session, client, ["mrv"], tmp_path)
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await ingest(db_session, client, ["mrv"], tmp_path)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] is not None
    assert versions[0] == versions[1]


async def test_changed_document_produces_a_new_corpus_version(db_session, tmp_path, corpus_client):
    client, _ = corpus_client({"mrv": MRV_SPARQL}, mrv_docs())
    first = await ingest(db_session, client, ["mrv"], tmp_path)

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
    second = await ingest(db_session, client, ["mrv"], tmp_path)

    versions = [(await db_session.get(IngestRun, r.run_id)).corpus_version for r in (first, second)]
    assert versions[0] != versions[1]


async def test_failed_run_is_not_stamped_with_a_corpus_version(db_session, tmp_path, corpus_client):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL}, mrv_docs({"32023R2449": httpx.Response(400, text="bad")})
    )
    report = await ingest(db_session, client, ["mrv"], tmp_path)

    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED
    assert run.corpus_version is None


async def test_malformed_sparql_payload_raises_discovery_error(db_session, tmp_path, corpus_client):
    client, _ = corpus_client({"mrv": httpx.Response(200, json={"unexpected": True})}, {})
    with pytest.raises(DiscoveryError, match="malformed"):
        await ingest(db_session, client, ["mrv"], tmp_path)


async def chunk_rows(session: AsyncSession, ref: str | None = None) -> list[DocumentChunk]:
    stmt = select(DocumentChunk).order_by(DocumentChunk.id)
    if ref is not None:
        stmt = stmt.where(DocumentChunk.ref == ref)
    return list(await session.scalars(stmt))


async def test_run_persists_chunks_for_every_document(db_session, tmp_path, corpus_client):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
            "32023R2449": httpx.Response(200, content=FUELEU_HTML.encode()),
        },
    )
    report = await ingest(db_session, client, ["mrv"], tmp_path)

    rows = await chunk_rows(db_session)
    assert report.ok
    assert report.chunks_added == len(rows) > 0
    assert {row.ref for row in rows} == {"32015R0757", "32023R2449"}
    assert {row.corpus_version for row in rows} == {report.corpus_version}


async def test_second_identical_run_adds_and_removes_nothing(db_session, tmp_path, corpus_client):
    docs = {
        "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
        "32023R2449": httpx.Response(200, content=FUELEU_HTML.encode()),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client, ["mrv"], tmp_path)
    before = {row.id for row in await chunk_rows(db_session)}

    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    second = await ingest(db_session, client, ["mrv"], tmp_path)

    assert (second.chunks_added, second.chunks_removed) == (0, 0)
    assert second.chunks_unchanged == len(before)
    assert {row.id for row in await chunk_rows(db_session)} == before


async def test_dropped_document_loses_its_chunks(db_session, tmp_path, corpus_client):
    docs = {
        "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
        "32023R2449": httpx.Response(200, content=FUELEU_HTML.encode()),
    }
    client, _ = corpus_client({"mrv": MRV_SPARQL}, docs)
    await ingest(db_session, client, ["mrv"], tmp_path)
    assert await chunk_rows(db_session, "32023R2449")

    only_seed = httpx.Response(200, json=payload(binding("32015R0757", force="1")))
    client, _ = corpus_client({"mrv": only_seed}, docs)
    report = await ingest(db_session, client, ["mrv"], tmp_path)

    assert report.dropped == ["32023R2449"]
    assert await chunk_rows(db_session, "32023R2449") == []
    assert await chunk_rows(db_session, "32015R0757")


async def test_unparseable_document_is_recorded_and_others_persist(
    db_session, tmp_path, corpus_client
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
            "32023R2449": httpx.Response(200, content=b"<html>not eur-lex</html>"),
        },
    )
    report = await ingest(db_session, client, ["mrv"], tmp_path)

    assert "32023R2449" in report.unparsed
    assert not report.ok
    assert await chunk_rows(db_session, "32015R0757")
    assert await chunk_rows(db_session, "32023R2449") == []
    run = await db_session.get(IngestRun, report.run_id)
    assert run.status is IngestRunStatus.FAILED


async def test_missing_source_file_is_recorded_not_raised(
    db_session, tmp_path, corpus_client, monkeypatch
):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
            "32023R2449": httpx.Response(200, content=FUELEU_HTML.encode()),
        },
    )
    monkeypatch.setattr(corpus, "store", lambda data_dir, ref, content: ("a" * 64, len(content)))
    report = await ingest(db_session, client, ["mrv"], tmp_path)

    assert sorted(report.unparsed) == ["32015R0757", "32023R2449"]
    assert all("FileNotFoundError" in reason for reason in report.unparsed.values())
    assert not report.ok


async def test_failed_fetch_still_chunks_what_was_downloaded(db_session, tmp_path, corpus_client):
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        {
            "32015R0757": httpx.Response(200, content=FUELEU_HTML.encode()),
            "32023R2449": httpx.Response(400, text="bad"),
        },
    )
    report = await ingest(db_session, client, ["mrv"], tmp_path)

    assert "32023R2449" in report.failed
    assert not report.ok
    assert report.corpus_version is None
    assert await chunk_rows(db_session, "32015R0757")
