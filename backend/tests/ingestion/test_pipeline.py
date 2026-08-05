"""The ingest orchestrator: run lifecycle and corpus versioning around the fetch stage."""

import re

import httpx
import pytest
from sqlalchemy import select

from app.ingestion.enums import IngestRunStatus
from app.ingestion.exceptions import DiscoveryError
from app.ingestion.pipeline import ingest
from app.ingestion.schemas import IngestRun
from tests.conftest import binding, payload

pytestmark = pytest.mark.anyio

MRV_SPARQL = httpx.Response(
    200, json=payload(binding("32015R0757", force="1"), binding("32023R2449", force="1"))
)


def mrv_docs(overrides: dict[str, httpx.Response] | None = None) -> dict[str, httpx.Response]:
    """The two-document mrv corpus, with per-ref responses overridable."""
    return {
        "32015R0757": httpx.Response(200, content=b"<html>mrv</html>"),
        "32023R2449": httpx.Response(200, content=b"<html>act</html>"),
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
    client, _ = corpus_client(
        {"mrv": MRV_SPARQL},
        mrv_docs({"32015R0757": httpx.Response(200, content=b"<html>v1</html>")}),
    )
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
        mrv_docs({"02015R0757-20250101": httpx.Response(200, content=b"<html>v2</html>")}),
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
