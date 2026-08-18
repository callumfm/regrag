"""Evals CLI: `uv run evals check`."""

import argparse
import asyncio

from app.core.db.session import get_session
from app.core.logger import setup_logging
from app.evals.models import EvalDataset, UnresolvedReference
from app.evals.service import find_unresolved_references


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="RegRag evals")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="confirm every case reference still resolves in the corpus")
    return parser


async def _check_dataset_references() -> tuple[UnresolvedReference, ...]:
    async with get_session(auto_commit=False) as session:
        return await find_unresolved_references(session, EvalDataset.load())


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    setup_logging()
    unresolved = asyncio.run(_check_dataset_references())
    for item in unresolved:
        print(f"{item.case_id}: no stored chunk for {item.target.celex} {item.target.citation}")
    if unresolved:
        return 1
    print("every case reference resolves")
    return 0
