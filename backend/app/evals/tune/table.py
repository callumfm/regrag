"""The tune run as one ranked table: recall against what it costs, deltas first."""

from app.evals.tune.models import TunedPoint, TuneRun
from app.evals.tune.service import tunable_fields

HEADER = (
    f"{'rank':>4}  {{label:<{{width}}}}  {'Δexp_rec':>8}  {'Δchunks':>7}  "
    f"{'exp_rec':>7}  {'raw_rec':>7}  {'refused':>7}  {'false':>5}  "
    f"{'chunks':>6}  {'chars':>7}  {'ms':>5}"
)


def _rate(value: float | None) -> str:
    """A rate as a fixed-width figure, or a dash holding its place when unmeasured."""
    return f"{value:>7.2f}" if value is not None else f"{'-':>7}"


def _delta(value: float | None, baseline: float | None, precision: int) -> str:
    """A signed movement against baseline, or a dash when either side is unmeasured."""
    if value is None or baseline is None:
        return "-"
    return f"{value - baseline:+.{precision}f}"


def _sort_key(result: TunedPoint) -> tuple[float, float]:
    """Best recall first; identical recall is won by the cheaper context."""
    recall = result.metrics.expanded_recall
    return (-(recall if recall is not None else -1.0), result.metrics.mean_context_chars or 0.0)


def _row(rank: int, result: TunedPoint, baseline: TunedPoint, width: int) -> str:
    m, b = result.metrics, baseline.metrics
    chars = f"{m.mean_context_chars / 1000:.1f}k" if m.mean_context_chars is not None else "-"
    chunks = f"{m.mean_context_chunks:.1f}" if m.mean_context_chunks is not None else "-"
    return (
        f"{rank:>4}  {result.point.label:<{width}}  "
        f"{_delta(m.expanded_recall, b.expanded_recall, 2):>8}  "
        f"{_delta(m.mean_context_chunks, b.mean_context_chunks, 1):>7}  "
        f"{_rate(m.expanded_recall)}  {_rate(m.raw_recall)}  "
        f"{_rate(m.gate_refusal_rate)}  {m.false_refusals:>5}  "
        f"{chunks:>6}  {chars:>7}  {m.mean_retrieve_ms:>5}"
    )


def format_tune_table(run: TuneRun) -> str:
    """Every point ranked, its movement against baseline first, then the absolutes;
    the baseline settings named below so the deltas can be read without the env."""
    ranked = sorted(run.results, key=_sort_key)
    width = max(len(result.point.label) for result in ranked)
    header = HEADER.format(label="point", width=width)
    rows = [_row(rank, result, run.baseline, width) for rank, result in enumerate(ranked, start=1)]
    footer = "baseline: " + " ".join(
        f"{name}={run.settings.root[name]}" for name in sorted(tunable_fields())
    )
    return "\n".join([header, *rows, "", footer])
