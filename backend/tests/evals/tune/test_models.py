"""Tune values: grid point labels and the run's shape."""

from app.evals.models import RunSettings
from app.evals.tune.models import GridPoint, RetrievalMetrics, TunedPoint, TuneRun

METRICS = RetrievalMetrics(
    cases=2,
    in_corpus=1,
    out_of_corpus=1,
    errors=0,
    raw_hit_rate=1.0,
    raw_recall=1.0,
    expanded_hit_rate=1.0,
    expanded_recall=1.0,
    gate_refusal_rate=1.0,
    false_refusals=0,
    refused_a_found_reference=0,
    mean_context_chunks=1.0,
    mean_context_chars=100.0,
    mean_retrieve_ms=100,
)


def test_the_empty_point_is_the_baseline() -> None:
    assert GridPoint().label == "(baseline)"


def test_a_point_labels_its_overrides_in_order() -> None:
    point = GridPoint(overrides={"CHAT_SOURCES": 8, "RERANK_ENABLED": False})

    assert point.label == "CHAT_SOURCES=8 RERANK_ENABLED=False"


def test_a_run_names_its_first_result_the_baseline() -> None:
    baseline = TunedPoint(point=GridPoint(), metrics=METRICS)
    varied = TunedPoint(point=GridPoint(overrides={"CHAT_SOURCES": 8}), metrics=METRICS)
    run = TuneRun(
        dataset_sha="a" * 64,
        case_pattern=None,
        settings=RunSettings.from_config(),
        results=(baseline, varied),
    )

    assert run.baseline is baseline
