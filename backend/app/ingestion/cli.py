"""Ingest CLI: `uv run ingest fetch [topics...]`."""

import argparse
import asyncio
import sys

import httpx

from app.core.db.session import get_session
from app.ingestion.discover import SEEDS, DiscoveryError
from app.ingestion.enums import DocAction
from app.ingestion.fetch import DATA_DIR, RunReport, fetch_topics

__all__ = ["DocAction", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description="RegRag corpus ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch", help="discover and download the corpus")
    fetch_parser.add_argument(
        "topics",
        nargs="*",
        metavar="topic",
        help=f"topics to fetch (default: {', '.join(sorted(SEEDS))})",
    )
    return parser


async def _fetch(topics: list[str]) -> RunReport:
    async with get_session(auto_commit=False) as session:
        return await fetch_topics(session, topics, DATA_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    topics = args.topics or sorted(SEEDS)
    unknown = sorted(set(topics) - SEEDS.keys())
    if unknown:
        parser.error(f"unknown topics: {', '.join(unknown)} (known: {', '.join(sorted(SEEDS))})")
    try:
        report = asyncio.run(_fetch(topics))
    except (DiscoveryError, httpx.HTTPError) as exc:
        print(f"fetch aborted: {exc}", file=sys.stderr)
        return 1
    print(report.summary())
    return 0 if report.ok else 1
