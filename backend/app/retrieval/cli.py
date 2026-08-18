"""Retrieval CLI: `uv run retrieve "query" [--celex C] [--topic T] [--limit N]`."""

import argparse
import asyncio
import sys

from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import LLMError
from app.core.logger import setup_logging
from app.retrieval.models import SearchFilters, SearchRequest, SearchResult
from app.retrieval.search import search

SNIPPET = 160


def _signal(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "-     "


def format_result(result: SearchResult) -> str:
    """One hit on one line: its three retrieval signals, then what and where it is."""
    return (
        f"rrf {result.rrf_score:.4f}  cos {_signal(result.cosine_similarity)}  "
        f"rel {_signal(result.reranker_relevance)}  {result.citation:<16} {result.celex}  "
        f"{result.text[:SNIPPET]}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieve", description="RegRag corpus retrieval")
    parser.add_argument("query", help="what to search the corpus for")
    parser.add_argument("--celex", help="restrict to one act")
    parser.add_argument("--topic", help="restrict to one topic")
    parser.add_argument(
        "--limit", type=int, help=f"results to return (default: {config.SEARCH_DEFAULT_LIMIT})"
    )
    return parser


async def _search(request: SearchRequest) -> tuple[SearchResult, ...]:
    async with get_session(auto_commit=False) as session:
        return await search(session, request)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    request = SearchRequest(
        query=args.query,
        filters=SearchFilters(celex=args.celex, topic=args.topic),
        limit=args.limit,
    )
    try:
        found = asyncio.run(_search(request))
    except LLMError as exc:
        print(f"retrieval failed: {exc}", file=sys.stderr)
        return 1
    for result in found:
        print(format_result(result))
    if not found:
        print("no results")
    return 0
