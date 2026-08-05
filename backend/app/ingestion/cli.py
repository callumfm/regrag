"""Ingest CLI: `uv run ingest [topics...]`."""

import argparse
import asyncio
import sys

import httpx

from app.core.config import config
from app.core.db.session import get_session
from app.core.http import http_client
from app.core.logger import setup_logging
from app.ingestion.constants import SEEDS
from app.ingestion.exceptions import DiscoveryError
from app.ingestion.models import RunReport
from app.ingestion.pipeline import ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description="RegRag corpus ingestion")
    parser.add_argument(
        "topics",
        nargs="*",
        metavar="topic",
        help=f"topics to ingest (default: {', '.join(sorted(SEEDS))})",
    )
    return parser


async def _ingest(topics: list[str]) -> RunReport:
    with http_client() as client:
        async with get_session(auto_commit=False) as session:
            return await ingest(session, client, topics, config.RAW_DATA_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    topics = args.topics or sorted(SEEDS)
    unknown = sorted(set(topics) - SEEDS.keys())
    if unknown:
        parser.error(f"unknown topics: {', '.join(unknown)} (known: {', '.join(sorted(SEEDS))})")
    setup_logging()
    try:
        report = asyncio.run(_ingest(topics))
    except (DiscoveryError, httpx.HTTPError) as exc:
        print(f"ingest aborted: {exc}", file=sys.stderr)
        return 1
    print(report.summary())
    return 0 if report.ok else 1
