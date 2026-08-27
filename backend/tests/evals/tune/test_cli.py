"""The tune subcommand: exit codes and what it prints."""

import pytest

from app.evals.cli import main
from app.evals.models import RunSettings
from app.evals.tune import cli as tune_cli
from app.evals.tune.models import GridPoint, TunedPoint, TuneRun
from app.evals.tune.params import TUNABLE_PARAMS
from tests.evals.tune.conftest import metrics


@pytest.fixture
def fake_grid(monkeypatch):
    """Replace the grid run with a stub returning a canned TuneRun, recording its points."""
    calls: dict = {}

    async def _fake(dataset, points, pattern=None):
        calls["points"] = points
        baseline = TunedPoint(point=GridPoint(), metrics=metrics())
        varied = tuple(TunedPoint(point=point, metrics=metrics()) for point in points)
        return TuneRun(
            dataset_sha=dataset.sha256,
            case_pattern=pattern,
            settings=RunSettings.from_config(),
            results=(baseline, *varied),
        )

    monkeypatch.setattr(tune_cli, "run_grid", _fake)
    return calls


def test_tune_sweeps_the_curated_params_and_prints_the_table(fake_grid, capsys):
    assert main(["tune"]) == 0

    varied = {name for point in fake_grid["points"] for name in point.overrides}
    assert varied
    assert varied <= set(TUNABLE_PARAMS)
    out = capsys.readouterr().out
    assert "(baseline)" in out
    assert "CHAT_SOURCES=3" in out
    assert "baseline:" in out


def test_tune_exits_nonzero_when_a_point_had_errors(monkeypatch, capsys):
    async def _fake(dataset, points, pattern=None):
        return TuneRun(
            dataset_sha=dataset.sha256,
            case_pattern=pattern,
            settings=RunSettings.from_config(),
            results=(TunedPoint(point=GridPoint(), metrics=metrics(errors=1)),),
        )

    monkeypatch.setattr(tune_cli, "run_grid", _fake)

    assert main(["tune"]) == 1
