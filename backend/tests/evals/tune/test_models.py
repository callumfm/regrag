"""Tune values: grid point labels and the run's shape."""

from app.evals.models import RunSettings
from app.evals.tune.models import GridPoint, TunedPoint, TuneRun
from tests.evals.tune.conftest import metrics


def test_the_empty_point_is_the_baseline() -> None:
    assert GridPoint().label == "(baseline)"


def test_a_point_labels_its_overrides_in_order() -> None:
    point = GridPoint(overrides={"CHAT_SOURCES": 8, "RERANK_ENABLED": False})

    assert point.label == "CHAT_SOURCES=8 RERANK_ENABLED=False"


def test_a_run_names_its_first_result_the_baseline() -> None:
    baseline = TunedPoint(point=GridPoint(), metrics=metrics())
    varied = TunedPoint(point=GridPoint(overrides={"CHAT_SOURCES": 8}), metrics=metrics())
    run = TuneRun(
        dataset_sha="a" * 64,
        case_pattern=None,
        settings=RunSettings.from_config(),
        results=(baseline, varied),
    )

    assert run.baseline is baseline
