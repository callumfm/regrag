"""The tune run as one ranked table: recall against what it costs, deltas first."""

from app.evals.tune.models import TunedPoint, TuneRun
from app.evals.tune.params import TUNABLE_PARAMS

HEADERS = (
    "rank",
    "point",
    "Δexp_rec",
    "Δchunks",
    "exp_rec",
    "raw_rec",
    "refused",
    "false",
    "chunks",
    "chars",
    "ms",
)

LABEL_COLUMN = HEADERS.index("point")
"""The one column that reads left-aligned: a name, not a figure."""


def _rate(value: float | None) -> str:
    """A rate as a figure, or a dash holding its place when unmeasured."""
    return f"{value:.2f}" if value is not None else "-"


def _delta(value: float | None, baseline: float | None, precision: int) -> str:
    """A signed movement against baseline, or a dash when either side is unmeasured."""
    if value is None or baseline is None:
        return "-"
    return f"{value - baseline:+.{precision}f}"


def _sort_key(result: TunedPoint) -> tuple[float, float]:
    """Best recall first; identical recall is won by the cheaper context."""
    recall = result.metrics.expanded_recall
    return (-(recall if recall is not None else -1.0), result.metrics.mean_context_chars or 0.0)


def _cells(rank: int, result: TunedPoint, baseline: TunedPoint) -> tuple[str, ...]:
    """One point's row, every measure already a string so the renderer only aligns."""
    m, b = result.metrics, baseline.metrics
    return (
        str(rank),
        result.point.label,
        _delta(m.expanded_recall, b.expanded_recall, 2),
        _delta(m.mean_context_chunks, b.mean_context_chunks, 1),
        _rate(m.expanded_recall),
        _rate(m.raw_recall),
        _rate(m.gate_refusal_rate),
        str(m.false_refusals),
        f"{m.mean_context_chunks:.1f}" if m.mean_context_chunks is not None else "-",
        f"{m.mean_context_chars / 1000:.1f}k" if m.mean_context_chars is not None else "-",
        str(m.mean_retrieve_ms),
    )


def _render(rows: list[tuple[str, ...]]) -> str:
    """Columns sized to their widest cell, figures right-aligned, the label left."""
    widths = [max(len(cell) for cell in column) for column in zip(*rows, strict=True)]
    return "\n".join(
        "  ".join(
            cell.ljust(width) if index == LABEL_COLUMN else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths, strict=True))
        ).rstrip()
        for row in rows
    )


def format_tune_table(run: TuneRun) -> str:
    """Every point ranked, its movement against baseline first, then the absolutes;
    the tunable params' baseline values named below so the deltas read without the env."""
    ranked = sorted(run.results, key=_sort_key)
    rows = [_cells(rank, result, run.baseline) for rank, result in enumerate(ranked, start=1)]
    footer = "baseline: " + " ".join(
        f"{name}={run.settings.root[name]}" for name in sorted(TUNABLE_PARAMS)
    )
    return "\n".join([_render([HEADERS, *rows]), "", footer])
