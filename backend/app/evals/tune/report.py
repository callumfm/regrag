"""The tune run as one ranked table: recall against what it costs, deltas first."""

from dataclasses import dataclass

from app.evals.metrics import format_rate
from app.evals.tune.models import TuneMetrics, TuneResult, TuneRun
from app.evals.tune.params import TUNABLE_PARAMS

NULL_CHAR: str = "-"


@dataclass(frozen=True)
class TuneRow:
    rank: str
    param: str
    value: str
    delta_exp_recall: str
    delta_chunks: str
    exp_recall: str
    raw_recall: str
    refusal_rate: str
    false_refusals: str
    chunks: str
    chars: str
    retrieve_ms: str


@dataclass(frozen=True)
class Column:
    header: str
    attribute: str
    align_left: bool


COLUMNS = (
    Column("rank", "rank", False),
    Column("param", "param", True),
    Column("value", "value", True),
    Column("Δexp_recall", "delta_exp_recall", False),
    Column("Δchunks", "delta_chunks", False),
    Column("exp_recall", "exp_recall", False),
    Column("raw_recall", "raw_recall", False),
    Column("refusal_rate", "refusal_rate", False),
    Column("false_refusals", "false_refusals", False),
    Column("chunks", "chunks", False),
    Column("chars", "chars", False),
    Column("retrieve_ms", "retrieve_ms", False),
)


def _format_delta(value: float | None, baseline: float | None, precision: int) -> str:
    """A signed movement against baseline, or a dash when either side is unmeasured."""
    if value is None or baseline is None:
        return NULL_CHAR
    return f"{value - baseline:+.{precision}f}"


def _format_context_chunks(value: float | None) -> str:
    """Format mean context size in chunks."""
    return f"{value:.1f}" if value is not None else NULL_CHAR


def _format_context_chars(value: float | None) -> str:
    """Format mean context size in thousands of characters."""
    return f"{value / 1000:.1f}k" if value is not None else NULL_CHAR


def _format_latency(value: float | None) -> str:
    """Format latency retrieval speed."""
    return str(value) if value is not None else NULL_CHAR


def _format_param(result: TuneResult) -> str:
    """The knob a row moved — with its companion settings, or the row would claim the
    knob alone made its delta."""
    if not result.requires:
        return result.param
    companions = " ".join(f"{name}={value}" for name, value in sorted(result.requires.items()))
    return f"{result.param}[{companions}]"


def _format_baseline(run: TuneRun) -> str:
    """Format the tunable parameter values used by the baseline."""
    names = sorted(param.name for param in TUNABLE_PARAMS)
    settings = (f"{name}={run.settings[name]}" for name in names)
    return "baseline: " + " ".join(settings)


def _format_run_metadata(run: TuneRun) -> str:
    """Format metadata identifying the tune run."""
    return (
        f"run: dataset_sha={run.dataset_sha[:12]} case_filter={run.case_filter} cached={run.cached}"
    )


def _sort_key(metrics: TuneMetrics) -> tuple[float, bool, float]:
    """Best recall first; identical recall is won by the cheaper context, and a result
    with no measured cost ranks after any measured one."""
    recall = metrics.expanded_recall
    context_chars = metrics.mean_context_chars

    recall_score = recall if recall is not None else -1.0
    has_no_context = context_chars is None
    context_size = context_chars if not has_no_context else 0.0

    return (-recall_score, has_no_context, context_size)


def _build_row(rank: int, param: str, value: str, m: TuneMetrics, b: TuneMetrics) -> TuneRow:
    """One result's row, every measure already a string so the renderer only aligns."""
    return TuneRow(
        rank=str(rank),
        param=param,
        value=value,
        delta_exp_recall=_format_delta(m.expanded_recall, b.expanded_recall, 2),
        delta_chunks=_format_delta(m.mean_context_chunks, b.mean_context_chunks, 1),
        exp_recall=format_rate(m.expanded_recall),
        raw_recall=format_rate(m.raw_recall),
        refusal_rate=format_rate(m.gate_refusal_rate),
        false_refusals=str(m.false_refusals),
        chunks=_format_context_chunks(m.mean_context_chunks),
        chars=_format_context_chars(m.mean_context_chars),
        retrieve_ms=_format_latency(m.mean_retrieve_ms),
    )


def _column_widths(rows: list[TuneRow]) -> list[int]:
    """Calculate the width required by each column."""
    return [
        max(
            len(column.header),
            *(len(getattr(row, column.attribute)) for row in rows),
        )
        for column in COLUMNS
    ]


def _render_row(row: TuneRow, widths: list[int]) -> str:
    """Render a row with labels left-aligned and figures right-aligned."""
    formatted_cells = []
    for column, width in zip(COLUMNS, widths, strict=True):
        cell = getattr(row, column.attribute)
        formatted_cell = cell.ljust(width) if column.align_left else cell.rjust(width)
        formatted_cells.append(formatted_cell)

    return "  ".join(formatted_cells).rstrip()


def _render_table(rows: list[TuneRow]) -> str:
    """Render rows as an aligned table."""
    header = TuneRow(**{column.attribute: column.header for column in COLUMNS})
    all_rows = [header, *rows]
    widths = _column_widths(all_rows)
    rendered_rows = (_render_row(row, widths) for row in all_rows)
    return "\n".join(rendered_rows)


def format_tune_table(run: TuneRun) -> str:
    """Format a tune run as a ranked table with baseline comparisons."""
    entries = [
        ("(baseline)", NULL_CHAR, run.baseline),
        *((_format_param(result), str(result.value), result.metrics) for result in run.results),
    ]
    ranked = sorted(entries, key=lambda entry: _sort_key(entry[2]))
    rows = [
        _build_row(rank, param, value, metrics, run.baseline)
        for rank, (param, value, metrics) in enumerate(ranked, start=1)
    ]

    table = _render_table(rows)
    baseline = _format_baseline(run)
    run_metadata = _format_run_metadata(run)

    output = [table, "", baseline, run_metadata]
    return "\n".join(output)
