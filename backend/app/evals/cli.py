"""Evals CLI: `uv run evals check`, `uv run evals run [--case PATTERN]`."""

import argparse
import asyncio

from app.core.db.session import get_session
from app.core.logger import setup_logging
from app.evals.models import EvalDataset, UnresolvedReference
from app.evals.service import find_unresolved_references, run_dataset


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


def check_references() -> int:
    """Name every case reference the corpus no longer answers to."""
    unresolved = asyncio.run(_check_dataset_references())
    for item in unresolved:
        print(f"{item.case_id}: no stored chunk for {item.target.celex} {item.target.citation}")
    if unresolved:
        return 1
    print("every case reference resolves")
    return 0


def run_evals(pattern: str | None) -> int:
    """Score the dataset and print what it measured."""
    run = asyncio.run(run_dataset(EvalDataset.load(), pattern))
    if not run.results:
        print(f"no case id contains {pattern!r}")
        return 1
    print(run.summary())
    return 1 if run.metrics.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    return run_evals(args.case) if args.command == "run" else check_references()
