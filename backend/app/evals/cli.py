"""Evals CLI: `uv run evals check`."""

import argparse
import asyncio

from app.core.db.session import get_session
from app.core.logger import setup_logging
from app.evals.dataset import load_golden
from app.evals.service import unresolved_references
from app.retrieval.models import ReferenceTarget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="RegRag evals")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="confirm every case reference still resolves in the corpus")
    return parser


async def _check() -> tuple[tuple[str, ReferenceTarget], ...]:
    async with get_session(auto_commit=False) as session:
        return await unresolved_references(session, load_golden())


def _describe(target: ReferenceTarget) -> str:
    division = f"Annex {target.annex}" if target.annex is not None else f"Article {target.article}"
    return f"{target.celex} {division}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    if args.command == "check":
        unresolved = asyncio.run(_check())
        for case_id, target in unresolved:
            print(f"{case_id}: no stored chunk for {_describe(target)}")
        if unresolved:
            return 1
        print("every case reference resolves")
    return 0
