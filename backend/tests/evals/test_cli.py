"""Evals CLI: exit codes and what `check` prints."""

import pytest

from app.evals import cli
from app.evals.cli import main
from app.evals.models import UnresolvedReference
from app.evals.results import RunResult
from app.retrieval.models import ReferenceTarget
from tests.evals.conftest import case_result


@pytest.fixture
def fake_check(monkeypatch):
    """Replace the DB coroutine with a stub returning a chosen set of unresolved references."""
    unresolved = []

    async def _fake():
        return tuple(unresolved)

    monkeypatch.setattr(cli, "_check_dataset_references", _fake)
    return unresolved


def test_check_exits_zero_when_every_reference_resolves(fake_check, capsys):
    assert main(["check"]) == 0
    assert "resolve" in capsys.readouterr().out


def test_check_names_each_stale_reference_and_exits_nonzero(fake_check, capsys):
    fake_check.append(
        UnresolvedReference(
            case_id="stale-case", target=ReferenceTarget(celex="32023R1805", article="999")
        )
    )

    assert main(["check"]) == 1

    out = capsys.readouterr().out
    assert "stale-case" in out
    assert "32023R1805" in out
    assert "999" in out


def test_a_subcommand_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])


@pytest.fixture
def fake_run(monkeypatch):
    """Replace the graph-driving coroutine with one returning chosen case results."""
    results: list = []

    async def _fake(dataset, pattern=None):
        return RunResult.from_results(results, dataset.sha256, pattern)

    monkeypatch.setattr(cli, "run_dataset", _fake)
    return results


def test_run_prints_the_settings_and_scores_it_measured(fake_run, capsys):
    fake_run.append(case_result())

    assert main(["run"]) == 0

    out = capsys.readouterr().out
    assert '"retrieval"' in out
    assert '"chat_model"' in out


def test_run_exits_nonzero_when_a_case_errored(fake_run, capsys):
    fake_run.append(case_result(error="provider down"))

    assert main(["run"]) == 1
    assert "provider down" in capsys.readouterr().out


def test_run_says_so_when_the_pattern_matches_no_case(fake_run, capsys):
    assert main(["run", "--case", "nope"]) == 1
    assert "nope" in capsys.readouterr().out
