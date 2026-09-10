"""Ingest CLI: argument validation, exit codes, report printing."""

import contextlib

import httpx
import pytest

from app.core.config import config
from app.core.llm.settings import MissingModelKeyError
from app.ingestion import cli
from app.ingestion.chunk.models import ChunkCounts
from app.ingestion.cli import main
from app.ingestion.enums import DocChange, Stage
from app.ingestion.exceptions import MalformedDiscoveryError
from app.ingestion.models import DocumentOutcome, IngestRunResult

pytestmark = pytest.mark.anyio


@pytest.fixture
def fake_ingest(monkeypatch):
    """Replace the DB+network coroutine with a stub recording requested topics."""
    calls = []
    report = IngestRunResult(run_id=1)

    async def _fake(topics):
        calls.append(list(topics))
        return report

    monkeypatch.setattr(cli, "_ingest", _fake)
    return calls, report


def test_no_topics_ingests_every_topic(fake_ingest):
    calls, _ = fake_ingest
    assert main([]) == 0
    assert calls == [sorted(config.TOPIC_BASE_ACTS)]


def test_explicit_topics_passed_through(fake_ingest):
    calls, _ = fake_ingest
    assert main(["mrv"]) == 0
    assert calls == [["mrv"]]


def test_unknown_topic_rejected(fake_ingest, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["shipping"])
    assert excinfo.value.code == 2
    assert "unknown topics: shipping" in capsys.readouterr().err


def test_prints_summary_and_exits_zero_when_clean(fake_ingest, capsys):
    _, report = fake_ingest
    report.documents.append(
        DocumentOutcome(celex="32023R1805", change=DocChange.NEW, chunks=ChunkCounts(added=3))
    )
    assert main([]) == 0
    assert "1 new" in capsys.readouterr().out


def test_exits_nonzero_when_documents_failed(fake_ingest, capsys):
    _, report = fake_ingest
    report.documents.append(
        DocumentOutcome(
            celex="32023R2917",
            failed=Stage.FETCH,
            error="ConnectionError: no fetchable HTML",
        )
    )
    assert main([]) == 1
    assert "fetch failed: 32023R2917" in capsys.readouterr().out


def test_exits_nonzero_when_a_document_failed_to_parse(fake_ingest, capsys):
    _, report = fake_ingest
    report.documents.append(
        DocumentOutcome(
            celex="32023R2449",
            failed=Stage.PARSE,
            error="ParseError: unrecognised dialect",
        )
    )
    assert main([]) == 1
    assert "parse failed: 32023R2449" in capsys.readouterr().out


def test_abort_prints_error_and_exits_nonzero(monkeypatch, capsys):
    async def _boom(topics):
        raise MalformedDiscoveryError("mrv: malformed SPARQL response")

    monkeypatch.setattr(cli, "_ingest", _boom)
    assert main([]) == 1
    assert "ingest aborted: mrv" in capsys.readouterr().err


def test_abort_on_http_error(monkeypatch, capsys):
    async def _boom(topics):
        raise httpx.ConnectError("endpoint down")

    monkeypatch.setattr(cli, "_ingest", _boom)
    assert main([]) == 1
    assert "ingest aborted" in capsys.readouterr().err


async def test_ingest_hands_the_pipeline_a_paced_client(monkeypatch):
    """The CLI is where the crawl delays are attached, so pin both the delays it asks for and
    that the real client it gets back carries the hook they produce."""
    built: dict = {}
    clients: list[httpx.AsyncClient] = []
    real_client = cli.http_client

    def spy(**kwargs):
        built.update(kwargs)
        return real_client(**kwargs)

    @contextlib.asynccontextmanager
    async def fake_session(**_):
        yield None

    async def capture(_session, *, client, **__):
        clients.append(client)
        return IngestRunResult(run_id=1)

    monkeypatch.setattr(cli, "http_client", spy)
    monkeypatch.setattr(cli, "get_session", fake_session)
    monkeypatch.setattr(cli, "ingest", capture)
    await cli._ingest(["mrv"])

    (client,) = clients
    assert built["delays"] == config.CRAWL_DELAYS
    assert client.event_hooks["request"]


def test_a_missing_embedding_key_stops_the_run_before_any_download(fake_ingest, monkeypatch):
    """A corpus crawl is slow and polite; discovering the key at the embed stage wastes it."""
    calls, _ = fake_ingest
    monkeypatch.setattr(config, "EMBED_MODEL", "openai/text-embedding-3-small")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingModelKeyError, match="OPENAI_API_KEY"):
        main([])

    assert calls == []


def test_ingest_never_asks_for_a_key_it_will_not_use(fake_ingest, monkeypatch):
    """It embeds and never answers, so a chat model at a provider with no key is its business
    to ignore — the point of checking per entry point rather than once for everything."""
    monkeypatch.setattr(config, "CHAT_MODEL", "openai/gpt-5")
    monkeypatch.setattr(config, "EVAL_JUDGE_MODEL", "openai/gpt-5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert main([]) == 0
