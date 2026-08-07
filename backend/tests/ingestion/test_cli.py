"""Ingest CLI: argument validation, exit codes, report printing."""

import httpx
import pytest

from app.ingestion import cli
from app.ingestion.cli import main
from app.ingestion.constants import SEEDS
from app.ingestion.enums import DocAction
from app.ingestion.exceptions import DiscoveryError, EmptyCorpusError
from app.ingestion.pipeline import IngestRunResult


@pytest.fixture
def fake_ingest(monkeypatch):
    """Replace the DB+network coroutine with a stub recording what each run was asked for."""
    calls = []
    report = IngestRunResult(run_id=1)

    async def _fake(topics, *, fetch=True):
        calls.append((list(topics), fetch))
        return report

    monkeypatch.setattr(cli, "_ingest", _fake)
    return calls, report


def test_no_topics_ingests_all_seeds(fake_ingest):
    calls, _ = fake_ingest
    assert main([]) == 0
    assert calls == [(sorted(SEEDS), True)]


def test_explicit_topics_passed_through(fake_ingest):
    calls, _ = fake_ingest
    assert main(["mrv"]) == 0
    assert calls == [(["mrv"], True)]


def test_no_fetch_reparses_the_stored_corpus(fake_ingest):
    calls, _ = fake_ingest
    assert main(["mrv", "--no-fetch"]) == 0
    assert calls == [(["mrv"], False)]


def test_unknown_topic_rejected(fake_ingest, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["shipping"])
    assert excinfo.value.code == 2
    assert "unknown topics: shipping" in capsys.readouterr().err


def test_prints_summary_and_exits_zero_when_clean(fake_ingest, capsys):
    _, report = fake_ingest
    report.fetch.record(DocAction.NEW, "32023R1805")
    assert main([]) == 0
    assert "1 new" in capsys.readouterr().out


def test_exits_nonzero_when_documents_failed(fake_ingest, capsys):
    _, report = fake_ingest
    report.fetch.failed["32023R2917"] = "ResolutionError: no fetchable HTML"
    assert main([]) == 1
    assert "fetch failed: 32023R2917" in capsys.readouterr().out


def test_exits_nonzero_when_a_document_failed_to_parse(fake_ingest, capsys):
    _, report = fake_ingest
    report.parse.failed["32023R2449"] = "ParseError: unrecognised EUR-Lex dialect"
    assert main([]) == 1
    assert "parse failed: 32023R2449" in capsys.readouterr().out


def test_abort_prints_error_and_exits_nonzero(monkeypatch, capsys):
    async def _boom(topics, *, fetch=True):
        raise DiscoveryError("mrv: malformed SPARQL response")

    monkeypatch.setattr(cli, "_ingest", _boom)
    assert main([]) == 1
    assert "ingest aborted: mrv" in capsys.readouterr().err


def test_abort_when_there_is_nothing_to_reparse(monkeypatch, capsys):
    async def _boom(topics, *, fetch=True):
        raise EmptyCorpusError("nothing fetched yet for: mrv")

    monkeypatch.setattr(cli, "_ingest", _boom)
    assert main(["mrv", "--no-fetch"]) == 1
    assert "ingest aborted: nothing fetched yet" in capsys.readouterr().err


def test_abort_on_http_error(monkeypatch, capsys):
    async def _boom(topics, *, fetch=True):
        raise httpx.ConnectError("endpoint down")

    monkeypatch.setattr(cli, "_ingest", _boom)
    assert main([]) == 1
    assert "ingest aborted" in capsys.readouterr().err
