"""Evals CLI: `uv run evals check`, `uv run evals run [--case PATTERN]`."""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from app.core.config import config
from app.core.db.session import get_session
from app.core.logger import setup_logging
from app.evals.models import EvalDataset, RunResult, UnresolvedReference
from app.evals.service import find_unresolved_references, run_dataset

RESULT_TIMESTAMP = "%Y%m%dT%H%M%SZ"
"""How a result file names the run it holds; sorts chronologically as a filename."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="RegRag evals")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="confirm every case reference still resolves in the corpus")
    run = commands.add_parser("run", help="score the dataset against the current chat graph")
    run.add_argument("--case", help="only cases whose id contains this")
    return parser


async def _check_dataset_references() -> tuple[UnresolvedReference, ...]:
    async with get_session(auto_commit=False) as session:
        return await find_unresolved_references(session, EvalDataset.load())


def check() -> int:
    """Name every case reference the corpus no longer answers to."""
    unresolved = asyncio.run(_check_dataset_references())
    for item in unresolved:
        print(f"{item.case_id}: no stored chunk for {item.target.celex} {item.target.citation}")
    if unresolved:
        return 1
    print("every case reference resolves")
    return 0


def result_path(started_at: datetime) -> Path:
    """Where a run started at this instant writes, kept clear of any file already there.

    Two runs can start inside one second — twenty cases that all fail fast take far less —
    and the earlier run's file is evidence, not something a retry may quietly overwrite.
    """
    directory = config.EVAL_RESULTS_DIR
    stem = started_at.strftime(RESULT_TIMESTAMP)
    path = directory / f"{stem}.json"
    attempt = 2
    while path.exists():
        path = directory / f"{stem}-{attempt}.json"
        attempt += 1
    return path


def write_result(run: RunResult) -> str:
    """The run as JSON under EVAL_RESULTS_DIR, named for when it started."""
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(run.started_at)
    path.write_text(run.model_dump_json(indent=2))
    return str(path)


def run_evals(pattern: str | None) -> int:
    """Score the dataset, print the table, and keep the run as a result file."""
    run = asyncio.run(run_dataset(EvalDataset.load(), pattern))
    if not run.results:
        print(f"no case id contains {pattern!r}")
        return 1
    print(run.table())
    print(f"\nwritten to {write_result(run)}")
    return 1 if run.metrics.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    return run_evals(args.case) if args.command == "run" else check()
