"""The tune table: ranked rows, deltas against baseline, and the baseline footer."""

from app.evals.models import RunSettings
from app.evals.tune.models import GridPoint, TunedPoint, TuneRun
from app.evals.tune.report import format_tune_table
from tests.evals.tune.conftest import metrics


def tune_run(*varied: TunedPoint) -> TuneRun:
    baseline = TunedPoint(point=GridPoint(), metrics=metrics())
    return TuneRun(
        dataset_sha="a" * 64,
        case_pattern=None,
        settings=RunSettings.from_config(),
        results=(baseline, *varied),
    )


def test_rows_rank_by_expanded_recall_then_cheaper_context() -> None:
    better = TunedPoint(
        point=GridPoint(overrides={"CHAT_SOURCES": 8}),
        metrics=metrics(expanded_recall=1.0, mean_context_chunks=17.1, mean_context_chars=34200.0),
    )
    worse = TunedPoint(
        point=GridPoint(overrides={"CHAT_SOURCES": 3}),
        metrics=metrics(expanded_recall=0.87, mean_context_chunks=10.4, mean_context_chars=20600.0),
    )
    same_but_cheaper = TunedPoint(
        point=GridPoint(overrides={"CHAT_CONTEXT_CHUNKS": 10}),
        metrics=metrics(mean_context_chunks=10.0, mean_context_chars=19800.0),
    )

    table = format_tune_table(tune_run(worse, same_but_cheaper, better))
    rows = table.splitlines()[1:5]

    assert rows[0].split()[1:3] == ["CHAT_SOURCES", "8"]
    assert rows[1].split()[1:3] == ["CHAT_CONTEXT_CHUNKS", "10"]
    assert rows[2].split()[1:3] == ["(baseline)", "-"]
    assert rows[3].split()[1:3] == ["CHAT_SOURCES", "3"]


def test_deltas_are_read_against_the_baseline() -> None:
    better = TunedPoint(
        point=GridPoint(overrides={"CHAT_SOURCES": 8}),
        metrics=metrics(expanded_recall=1.0, mean_context_chunks=17.1),
    )

    table = format_tune_table(tune_run(better))

    assert "+0.03" in table
    assert "+2.9" in table


def test_the_footer_names_every_tunable_baseline_value() -> None:
    table = format_tune_table(tune_run())
    footer = table.splitlines()[-1]

    assert footer.startswith("baseline:")
    assert "CHAT_SOURCES=" in footer
    assert "MIN_COSINE_SIMILARITY=" in footer
