"""Evals CLI: exit codes and what `check` and `run` print."""

import pytest

from app.evals import cli
from app.evals.cli import main
from app.evals.metrics import compute_metrics
from app.evals.models import EvalRun, RunSettings, UnresolvedReference
from app.retrieval.models import ReferenceTarget
from tests.evals.conftest import eval_case, eval_result


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
    """Replace the graph run with a stub returning a chosen list of results."""
    results: list = []

    async def _fake(dataset, pattern):
        chosen = tuple(results)
        return EvalRun(
            dataset_sha=dataset.sha256,
            case_pattern=pattern,
            settings=RunSettings.from_config(),
            metrics=compute_metrics(chosen),
            results=chosen,
        )

    monkeypatch.setattr(cli, "run_dataset", _fake)
    return results


def test_run_prints_the_summary_and_exits_zero(fake_run, capsys):
    fake_run.append(eval_result())

    assert main(["run"]) == 0

    out = capsys.readouterr().out
    assert '"raw_recall": 1.0' in out
    assert '"chat_model"' in out


def test_run_exits_nonzero_when_a_case_raised(fake_run, capsys):
    fake_run.append(eval_result(eval_case(id="boom"), error="TimeoutError"))

    assert main(["run"]) == 1
    assert "boom  TimeoutError" in capsys.readouterr().out


def test_run_exits_nonzero_when_no_case_matches_the_pattern(fake_run, capsys):
    assert main(["run", "--case", "nothing-here"]) == 1
    assert "nothing-here" in capsys.readouterr().out
