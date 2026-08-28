"""Tune CLI: the `evals tune` subcommand, from the curated params to the printed table."""

import asyncio
from typing import Any

from app.evals.cache import enable_call_cache
from app.evals.models import EvalDataset
from app.evals.tune.params import TUNABLE_PARAMS
from app.evals.tune.report import format_tune_table
from app.evals.tune.service import tune


def register_tune_command(commands: Any) -> None:
    """The tune subparser: sweep every curated param at its values, one at a time."""
    tune_parser = commands.add_parser(
        "tune",
        help="rank retrieval settings against the dataset",
        description="Sweep every param in TUNABLE_PARAMS at its curated values, one factor "
        "at a time, and print one ranked table. Edit params.py to change what is swept. "
        "Embed and rerank calls replay from the cache, so a sweep is Postgres time, not spend.",
    )
    tune_parser.add_argument("--case", help="only cases whose id contains this")
    tune_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="pay for every embed and rerank again instead of replaying the cached ones",
    )


def run_tune(case_filter: str | None, cached: bool = True) -> int:
    """Sweep the curated grid and print the ranked table."""
    dataset = EvalDataset.load(case_filter=case_filter)

    if cached:
        enable_call_cache()

    run = asyncio.run(tune(dataset, params=TUNABLE_PARAMS))
    print(format_tune_table(run))

    measured = (run.baseline, *(result.metrics for result in run.results))
    return 1 if any(metrics.errors for metrics in measured) else 0
