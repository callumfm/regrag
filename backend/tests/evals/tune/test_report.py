"""The tune table: ranked rows, deltas against baseline, and the footers."""

from app.evals.tune.models import TuneResult
from app.evals.tune.report import format_tune_table
from tests.evals.tune.conftest import metrics, tune_run


def test_rows_rank_by_expanded_recall_then_cheaper_context() -> None:
    better = TuneResult(
        param="CHAT_SOURCES",
        value=8,
        metrics=metrics(expanded_recall=1.0, mean_context_chunks=17.1, mean_context_chars=34200.0),
    )
    worse = TuneResult(
        param="CHAT_SOURCES",
        value=3,
        metrics=metrics(expanded_recall=0.87, mean_context_chunks=10.4, mean_context_chars=20600.0),
    )
    same_but_cheaper = TuneResult(
        param="CHAT_CONTEXT_CHUNKS",
        value=10,
        metrics=metrics(mean_context_chunks=10.0, mean_context_chars=19800.0),
    )

    table = format_tune_table(tune_run(worse, same_but_cheaper, better))
    rows = table.splitlines()[1:5]

    assert rows[0].split()[1:3] == ["CHAT_SOURCES", "8"]
    assert rows[1].split()[1:3] == ["CHAT_CONTEXT_CHUNKS", "10"]
    assert rows[2].split()[1:3] == ["(baseline)", "-"]
    assert rows[3].split()[1:3] == ["CHAT_SOURCES", "3"]


def test_an_unmeasured_context_cost_ranks_after_any_measured_one() -> None:
    """A result every case errored under measures nothing; nothing must not read as cheap."""
    unmeasured = TuneResult(
        param="RERANK_ENABLED",
        value=False,
        metrics=metrics(mean_context_chunks=None, mean_context_chars=None, mean_retrieve_ms=None),
    )

    table = format_tune_table(tune_run(unmeasured))
    rows = table.splitlines()[1:3]

    assert rows[0].split()[1] == "(baseline)"
    assert rows[1].split()[1] == "RERANK_ENABLED"
    assert rows[1].rstrip().endswith("-")


def test_deltas_are_read_against_the_baseline() -> None:
    better = TuneResult(
        param="CHAT_SOURCES",
        value=8,
        metrics=metrics(expanded_recall=1.0, mean_context_chunks=17.1),
    )

    table = format_tune_table(tune_run(better))

    assert "+0.03" in table
    assert "+2.9" in table


def test_the_footer_names_every_tunable_baseline_value_and_the_run_line() -> None:
    table = format_tune_table(tune_run())
    baseline, run_line = table.splitlines()[-2:]

    assert baseline.startswith("baseline:")
    assert "CHAT_SOURCES=" in baseline
    assert "MIN_COSINE_SIMILARITY=" in baseline
    assert run_line.startswith("run:")
    assert "dataset_sha=aaaaaaaaaaaa" in run_line
    assert "cached=False" in run_line


def test_a_gated_row_names_the_companions_it_ran_with() -> None:
    """The row changed two settings; a bare param name would claim it changed one."""
    gated = TuneResult(
        param="CHAT_CONTEXT_CHUNKS",
        value=10,
        requires={"EXPAND_SECTIONS": True},
        metrics=metrics(),
    )

    table = format_tune_table(tune_run(gated))

    assert "CHAT_CONTEXT_CHUNKS[EXPAND_SECTIONS=True]" in table
