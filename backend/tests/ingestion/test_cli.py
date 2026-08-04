"""Ingest CLI: argument validation, exit codes, report printing."""

import httpx
import pytest

from app.ingestion import cli
from app.ingestion.cli import main
from app.ingestion.enums import DocAction
from app.ingestion.fetch.discover import SEEDS, DiscoveryError
from app.ingestion.fetch.download import RunReport


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the DB+network coroutine with a stub recording requested topics."""
    calls = []
    report = RunReport(run_id=1)

    async def _fake(topics):
        calls.append(list(topics))
        return report

    monkeypatch.setattr(cli, "_fetch", _fake)
    return calls, report


def test_no_topics_fetches_all_seeds(fake_fetch):
    calls, _ = fake_fetch
    assert main(["fetch"]) == 0
    assert calls == [sorted(SEEDS)]


def test_explicit_topics_passed_through(fake_fetch):
    calls, _ = fake_fetch
    assert main(["fetch", "mrv"]) == 0
    assert calls == [["mrv"]]


def test_unknown_topic_rejected(fake_fetch, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["fetch", "shipping"])
    assert excinfo.value.code == 2
    assert "unknown topics: shipping" in capsys.readouterr().err


def test_missing_subcommand_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_prints_summary_and_exits_zero_when_clean(fake_fetch, capsys):
    _, report = fake_fetch
    report.record(DocAction.NEW, "32023R1805")
    assert main(["fetch"]) == 0
    assert "1 new" in capsys.readouterr().out


def test_exits_nonzero_when_documents_failed(fake_fetch, capsys):
    _, report = fake_fetch
    report.failed["32023R2917"] = "ResolutionError: no fetchable HTML"
    assert main(["fetch"]) == 1
    assert "failed: 32023R2917" in capsys.readouterr().out


def test_abort_prints_error_and_exits_nonzero(monkeypatch, capsys):
    async def _boom(topics):
        raise DiscoveryError("mrv: malformed SPARQL response")

    monkeypatch.setattr(cli, "_fetch", _boom)
    assert main(["fetch"]) == 1
    assert "fetch aborted: mrv" in capsys.readouterr().err


def test_abort_on_http_error(monkeypatch, capsys):
    async def _boom(topics):
        raise httpx.ConnectError("endpoint down")

    monkeypatch.setattr(cli, "_fetch", _boom)
    assert main(["fetch"]) == 1
    assert "fetch aborted" in capsys.readouterr().err
