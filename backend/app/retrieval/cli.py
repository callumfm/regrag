"""Retrieval CLI: `uv run retrieve "query" [--celex C] [--topic T] [--limit N]`."""

import argparse
import asyncio
import sys

from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import LLMError
from app.core.logger import setup_logging
from app.retrieval.models import SearchFilters, SearchResult
from app.retrieval.pipeline import search

SNIPPET = 160


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieve", description="RegRag corpus retrieval")
    parser.add_argument("query", help="what to search the corpus for")
    parser.add_argument("--celex", help="restrict to one act")
    parser.add_argument("--topic", help="restrict to one topic")
    parser.add_argument(
        "--limit", type=int, default=config.SEARCH_DEFAULT_LIMIT, help="results to return"
    )
    return parser


async def _search(query: str, filters: SearchFilters, limit: int) -> tuple[SearchResult, ...]:
    async with get_session(auto_commit=False) as session:
        return await search(session, query, filters, limit=limit)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    filters = SearchFilters(celex=args.celex, topic=args.topic)
    try:
        found = asyncio.run(_search(args.query, filters, args.limit))
    except LLMError as exc:
        print(f"retrieval failed: {exc}", file=sys.stderr)
        return 1
    for result in found:
        print(f"{result.score:.4f}  {result.citation:<16} {result.celex}  {result.text[:SNIPPET]}")
    if not found:
        print("no results")
    return 0
