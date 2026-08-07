"""Ingest CLI: `uv run ingest [topics...] [--no-fetch]`."""

import argparse
import asyncio
import sys

import httpx

from app.core.db.session import get_session
from app.core.http import http_client
from app.core.logger import setup_logging
from app.core.storage import get_object_store
from app.ingestion.constants import SEEDS
from app.ingestion.exceptions import IngestionError
from app.ingestion.pipeline import IngestRunResult, ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description="RegRag corpus ingestion")
    parser.add_argument(
        "topics",
        nargs="*",
        metavar="topic",
        help=f"topics to ingest (default: {', '.join(sorted(SEEDS))})",
    )
    parser.add_argument(
        "--fetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download from the source; --no-fetch reparses the stored corpus offline",
    )
    return parser


async def _ingest(topics: list[str], *, fetch: bool = True) -> IngestRunResult:
    with http_client() as client:
        async with get_session(auto_commit=False) as session:
            return await ingest(
                session, client=client, topics=topics, store=get_object_store(), fetch=fetch
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    topics = args.topics or sorted(SEEDS)
    unknown = sorted(set(topics) - SEEDS.keys())
    if unknown:
        parser.error(f"unknown topics: {', '.join(unknown)} (known: {', '.join(sorted(SEEDS))})")
    setup_logging()
    try:
        report = asyncio.run(_ingest(topics, fetch=args.fetch))
    except (IngestionError, httpx.HTTPError) as exc:
        print(f"ingest aborted: {exc}", file=sys.stderr)
        return 1
    print(report.summary())
    return 0 if report.ok else 1
