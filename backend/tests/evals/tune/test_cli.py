"""The tune subcommand: exit codes, cache handling, and what it prints."""

import pytest

from app.core.config import EVAL_CONFIG_SECTIONS, get_config_snapshot
from app.evals.cli import main
from app.evals.tune import cli as tune_cli
from app.evals.tune.models import TuneResult, TuneRun
from app.evals.tune.params import TUNABLE_PARAMS
from tests.evals.tune.conftest import metrics


@pytest.fixture
def fake_tune(monkeypatch):
    """Replace the sweep with a stub returning a canned TuneRun, recording its params."""
    calls: dict = {}

    async def _fake(dataset, params):
        calls["params"] = tuple(params)
        results = tuple(
            TuneResult(param=param.name, value=param.values[0], metrics=metrics())
            for param in params
        )
        return TuneRun(
            dataset_sha=dataset.sha256,
            selection=dataset.selection,
            settings=get_config_snapshot(EVAL_CONFIG_SECTIONS),
            baseline=metrics(),
            results=results,
        )

    monkeypatch.setattr(tune_cli, "tune", _fake)
    return calls


def test_tune_sweeps_the_curated_params_and_prints_the_table(fake_tune, capsys):
    assert main(["tune"]) == 0

    assert fake_tune["params"] == TUNABLE_PARAMS
    out = capsys.readouterr().out
    assert "(baseline)" in out
    assert "CHAT_SOURCES" in out
    assert "baseline:" in out


def test_tune_exits_nonzero_when_a_measurement_had_errors(monkeypatch, capsys):
    async def _fake(dataset, params):
        return TuneRun(
            dataset_sha=dataset.sha256,
            settings=get_config_snapshot(EVAL_CONFIG_SECTIONS),
            baseline=metrics(errors=1),
        )

    monkeypatch.setattr(tune_cli, "tune", _fake)

    assert main(["tune"]) == 1


def test_tune_replays_its_embed_and_rerank_calls_by_default(fake_tune, enabled):
    assert main(["tune"]) == 0
    assert enabled


def test_no_cache_makes_a_sweep_pay_for_its_calls_again(fake_tune, enabled):
    assert main(["tune", "--no-cache"]) == 0
    assert not enabled


def test_tune_names_an_empty_selection_before_any_sweep(fake_tune, enabled, capsys):
    assert main(["tune", "--case", "nothing-here"]) == 1

    assert "nothing-here" in capsys.readouterr().out
    assert "params" not in fake_tune
    assert not enabled


def test_tune_sweeps_the_cases_carrying_a_trait_and_records_the_filter(fake_tune, capsys):
    assert main(["tune", "--trait", "multi_hop"]) == 0
    assert "selection=trait=multi_hop" in capsys.readouterr().out
