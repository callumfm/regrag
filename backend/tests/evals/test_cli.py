"""Evals CLI: exit codes and what `check` prints."""

import json
from datetime import UTC, datetime

import pytest

from app.core.config import config
from app.evals import cli
from app.evals.cli import main
from app.evals.models import RunResult, UnresolvedReference
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
def fake_run(monkeypatch, tmp_path):
    """Replace the graph-driving coroutine with one returning chosen case results,
    and point result files at a temporary directory."""
    monkeypatch.setattr(config, "EVAL_RESULTS_DIR", tmp_path)
    results: list = []

    async def _fake(dataset, pattern=None):
        return RunResult.from_results(results, dataset.sha256)

    monkeypatch.setattr(cli, "run_dataset", _fake)
    return results


def test_run_prints_the_table_and_writes_a_result_file(fake_run, tmp_path, capsys):
    fake_run.append(case_result())

    assert main(["run"]) == 0

    out = capsys.readouterr().out
    assert "hit-rate@" in out
    assert "cited references" in out
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["metrics"]["expanded_hit_rate"] == 1.0


def test_run_records_the_settings_the_scores_were_produced_under(fake_run, tmp_path):
    fake_run.append(case_result())

    main(["run"])

    settings = json.loads(next(tmp_path.glob("*.json")).read_text())["settings"]
    assert settings["chat_model"] == config.CHAT_MODEL
    assert settings["min_cosine_similarity"] == config.MIN_COSINE_SIMILARITY


def test_run_exits_nonzero_when_a_case_errored(fake_run, capsys):
    fake_run.append(case_result(error="provider down"))

    assert main(["run"]) == 1


def test_run_says_so_when_the_pattern_matches_no_case(fake_run, capsys):
    assert main(["run", "--case", "nope"]) == 1
    assert "nope" in capsys.readouterr().out


def test_a_second_run_in_the_same_second_does_not_overwrite_the_first(fake_run, tmp_path):
    """Twenty cases that all fail fast take well under a second, and the earlier run's file
    is evidence rather than something a retry may quietly replace."""
    fake_run.append(case_result())

    main(["run"])
    main(["run"])

    written = {path.name for path in tmp_path.glob("*.json")}
    assert len(written) == 2
    assert any(name.endswith("-2.json") for name in written)


def test_a_result_file_is_named_for_the_instant_its_run_started(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EVAL_RESULTS_DIR", tmp_path)
    started = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)

    path = cli.result_path(started)

    assert path.name == "20260818T120000Z.json"
