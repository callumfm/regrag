"""Tune CLI: the `evals tune` subcommand, from flags to the printed table."""

import asyncio
import sys
from typing import Any

from pydantic import ValidationError

from app.evals.cache import enable_call_cache
from app.evals.models import EvalDataset
from app.evals.tune.grid import build_points, parse_param_values
from app.evals.tune.service import run_grid
from app.evals.tune.table import format_tune_table


def register_tune_command(commands: Any) -> None:
    """The tune subparser: sweep every curated param, or just the ones named."""
    tune = commands.add_parser(
        "tune",
        help="rank retrieval settings against the dataset",
        description="Rank retrieval settings against the dataset. With no --set, sweeps "
        "every param in TUNABLE_PARAMS at its curated values, one factor at a time. "
        "The curated params replay their embed and rerank calls from the cache for free.",
    )
    tune.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help="values to try for one param; repeatable; omit to sweep every curated param",
    )
    tune.add_argument(
        "--cross", action="store_true", help="cross every param instead of varying one at a time"
    )
    tune.add_argument("--case", help="only cases whose id contains this")


def run_tune(sets: list[str], cross: bool, pattern: str | None) -> int:
    """Validate the grid, run it with the call cache on, and print the ranked table."""
    try:
        points = build_points(parse_param_values(sets), cross=cross)
    except (ValueError, ValidationError) as exc:
        print(f"evals tune: error: {exc}", file=sys.stderr)
        return 2
    enable_call_cache()
    run = asyncio.run(run_grid(EvalDataset.load(), points, pattern))
    if run.baseline.metrics.cases == 0:
        print(f"no case id contains {pattern!r}")
        return 1
    print(format_tune_table(run))
    return 1 if any(result.metrics.errors for result in run.results) else 0
