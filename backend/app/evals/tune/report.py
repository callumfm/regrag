"""The tune run as one ranked table: recall against what it costs, deltas first."""

from collections.abc import Callable
from dataclasses import dataclass

from app.chat.enums import ChatNode
from app.evals.models import EvalMetrics
from app.evals.report import UNMEASURED, format_rate
from app.evals.tune.models import TuneResult, TuneRun
from app.evals.tune.params import TUNABLE_PARAMS


@dataclass(frozen=True)
class RankedEntry:
    """One row's subject: the baseline or a tried value, already placed in the ranking."""

    rank: int
    param: str
    value: str
    metrics: EvalMetrics


@dataclass(frozen=True)
class Column:
    """A header and the cell it prints for an entry, read against the baseline."""

    header: str
    cell: Callable[[RankedEntry, EvalMetrics], str]
    align_left: bool = False


def _format_delta(value: float | None, baseline: float | None, precision: int) -> str:
    """A signed movement against baseline, or a dash when either side is unmeasured."""
    if value is None or baseline is None:
        return UNMEASURED
    return f"{value - baseline:+.{precision}f}"


def _format_chunks(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else UNMEASURED


def _format_chars(value: float | None) -> str:
    """Thousands of characters."""
    return f"{value / 1000:.1f}k" if value is not None else UNMEASURED


def _format_ms(value: int | None) -> str:
    return str(value) if value is not None else UNMEASURED


COLUMNS = (
    Column("rank", lambda e, b: str(e.rank)),
    Column("param", lambda e, b: e.param, align_left=True),
    Column("value", lambda e, b: e.value, align_left=True),
    Column(
        "Δexp_recall",
        lambda e, b: _format_delta(
            e.metrics.retrieval.expanded_recall, b.retrieval.expanded_recall, 2
        ),
    ),
    Column(
        "Δchunks",
        lambda e, b: _format_delta(
            e.metrics.context.mean_context_chunks, b.context.mean_context_chunks, 1
        ),
    ),
    Column("exp_recall", lambda e, b: format_rate(e.metrics.retrieval.expanded_recall)),
    Column("raw_recall", lambda e, b: format_rate(e.metrics.retrieval.raw_recall)),
    Column("refusal_rate", lambda e, b: format_rate(e.metrics.gate.refusal_rate)),
    Column("false_refusals", lambda e, b: str(e.metrics.gate.false_refusals)),
    Column("chunks", lambda e, b: _format_chunks(e.metrics.context.mean_context_chunks)),
    Column("chars", lambda e, b: _format_chars(e.metrics.context.mean_context_chars)),
    Column(
        "retrieve_ms",
        lambda e, b: _format_ms(e.metrics.latency.mean_step_ms.get(ChatNode.RETRIEVE.value)),
    ),
)


def _format_param(result: TuneResult) -> str:
    """The knob a row moved — with its companion settings, or the row would claim the
    knob alone made its delta."""
    if not result.requires:
        return result.param
    companions = " ".join(f"{name}={value}" for name, value in sorted(result.requires.items()))
    return f"{result.param}[{companions}]"


def _format_baseline(run: TuneRun) -> str:
    """The tunable parameter values the baseline ran with."""
    names = sorted(param.name for param in TUNABLE_PARAMS)
    settings = (f"{name}={run.settings[name]}" for name in names)
    return "baseline: " + " ".join(settings)


def _format_run_metadata(run: TuneRun) -> str:
    return (
        f"run: dataset_sha={run.dataset_sha[:12]} selection={run.selection.describe()} "
        f"cached={run.cached}"
    )


def _sort_key(metrics: EvalMetrics) -> tuple[float, bool, float]:
    """Best recall first; identical recall is won by the cheaper context, and a result
    with no measured cost ranks after any measured one."""
    recall = metrics.retrieval.expanded_recall
    context_chars = metrics.context.mean_context_chars

    recall_score = recall if recall is not None else -1.0
    has_no_context = context_chars is None
    context_size = context_chars if not has_no_context else 0.0

    return (-recall_score, has_no_context, context_size)


def _rank(run: TuneRun) -> list[RankedEntry]:
    """The baseline and every tried value, best first."""
    entries = [
        ("(baseline)", UNMEASURED, run.baseline),
        *((_format_param(result), str(result.value), result.metrics) for result in run.results),
    ]
    ranked = sorted(entries, key=lambda entry: _sort_key(entry[2]))
    return [
        RankedEntry(rank, param, value, metrics)
        for rank, (param, value, metrics) in enumerate(ranked, start=1)
    ]


def _render_table(rows: list[list[str]]) -> str:
    """Header then rows, each column as wide as its widest cell: labels left, figures right."""
    header = [column.header for column in COLUMNS]
    widths = [max(len(cell) for cell in cells) for cells in zip(header, *rows, strict=True)]

    def render(row: list[str]) -> str:
        cells = (
            cell.ljust(width) if column.align_left else cell.rjust(width)
            for column, cell, width in zip(COLUMNS, row, widths, strict=True)
        )
        return "  ".join(cells).rstrip()

    return "\n".join(render(row) for row in [header, *rows])


def format_tune_table(run: TuneRun) -> str:
    """The ranked table, then the baseline's settings and the run line."""
    rows = [[column.cell(entry, run.baseline) for column in COLUMNS] for entry in _rank(run)]
    return "\n".join([_render_table(rows), "", _format_baseline(run), _format_run_metadata(run)])
