"""Tune CLI: the `evals tune` subcommand, from the curated params to the printed table."""

import asyncio
from typing import Any

from app.evals.cache import enable_call_cache
from app.evals.models import EvalDataset
from app.evals.tune.params import get_tunable_params
from app.evals.tune.report import format_tune_table
from app.evals.tune.service import build_grid, run_grid


def register_tune_command(commands: Any) -> None:
    """The tune subparser: sweep every curated param at its values, one at a time."""
    tune = commands.add_parser(
        "tune",
        help="rank retrieval settings against the dataset",
        description="Sweep every param in TUNABLE_PARAMS at its curated values, one factor "
        "at a time, and print one ranked table. Edit params.py to change what is swept. "
        "Embed and rerank calls replay from the cache, so a sweep is Postgres time, not spend.",
    )
    tune.add_argument("--case", help="only cases whose id contains this")


def run_tune(pattern: str | None) -> int:
    """Sweep the curated grid with the call cache on and print the ranked table."""
    points = build_grid(get_tunable_params())
    enable_call_cache()
    run = asyncio.run(run_grid(EvalDataset.load(), points, pattern))
    if run.baseline.metrics.cases == 0:
        print(f"no case id contains {pattern!r}")
        return 1
    print(format_tune_table(run))
    return 1 if any(result.metrics.errors for result in run.results) else 0
