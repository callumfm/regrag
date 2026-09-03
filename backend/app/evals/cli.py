"""Evals CLI: `uv run evals check | stamp | run [--case PATTERN] [--verbose] | tune`."""

import argparse
import asyncio
from typing import Any

from app.core.logger import setup_logging
from app.evals.cache import enable_call_cache
from app.evals.dataset.check import check_against_corpus, stale_case_ids
from app.evals.dataset.cli import register_dataset_commands, run_check, run_stamp
from app.evals.dataset.exceptions import DatasetError
from app.evals.dataset.models import EvalDataset
from app.evals.report import format_case_lines
from app.evals.service import evaluate_all_cases
from app.evals.tune.cli import register_tune_command, run_tune


def register_run_command(commands: Any) -> None:
    """The run subparser: drive the selected cases through the graph and print one summary."""
    run = commands.add_parser(
        "run",
        help="score the dataset against the current chat graph",
        description="Drive every selected case through the chat graph and print the run "
        "summary: which corpus and settings it scored against, what it measured, then any "
        "case owed a re-review or that raised. A stale case is reported, never failed.",
    )
    run.add_argument("--case", help="only cases whose id contains this")
    run.add_argument("--verbose", action="store_true", help="list every case with its own scores")
    run.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the LLM judge, leaving the judged metrics unmeasured",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="pay for every embed and rerank again instead of replaying the cached ones",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="RegRag evals")
    commands = parser.add_subparsers(dest="command", required=True)
    register_dataset_commands(commands)
    register_run_command(commands)
    register_tune_command(commands)
    return parser


def run_evals(
    case_filter: str | None, verbose: bool = False, cached: bool = True, judge: bool = True
) -> int:
    """Score the dataset and print what it measured, the cases first when asked for."""
    dataset = EvalDataset.load(case_filter=case_filter)
    drifted, corpus_version = asyncio.run(check_against_corpus(dataset))

    if cached:
        enable_call_cache()

    run = asyncio.run(
        evaluate_all_cases(dataset, corpus_version, stale_case_ids(drifted), judge=judge)
    )
    if verbose:
        print("\n".join(format_case_lines(run.results)), end="\n\n")

    print(run.summary())
    return 1 if run.metrics.counts.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    try:
        if args.command == "run":
            return run_evals(
                args.case, args.verbose, cached=not args.no_cache, judge=not args.no_judge
            )

        if args.command == "tune":
            return run_tune(args.case, cached=not args.no_cache)

        if args.command == "stamp":
            return run_stamp(args.case)

        return run_check()
    except DatasetError as exc:
        print(exc)
        return 1
