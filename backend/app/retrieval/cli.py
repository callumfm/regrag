"""Retrieval CLI: `uv run retrieve "query" [--topic t]` or `uv run retrieve --article CELEX N`."""

import argparse
import asyncio
import sys

from app.core.db.session import get_session
from app.core.llm import LLMError
from app.core.logger import setup_logging
from app.retrieval.constants import DEFAULT_LIMIT
from app.retrieval.models import RetrievedChunk, SearchFilters, SearchResult
from app.retrieval.pipeline import search
from app.retrieval.service import get_article

SNIPPET = 160


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieve", description="RegRag corpus retrieval")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("query", nargs="?", help="what to search the corpus for")
    target.add_argument(
        "--article", nargs=2, metavar=("CELEX", "ARTICLE"), help="look one article up exactly"
    )
    parser.add_argument("--celex", help="restrict to one act")
    parser.add_argument("--topic", help="restrict to one topic")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="results to return")
    return parser


def _rank(rank: int | None) -> str:
    """A leg's rank, or a dash where that leg did not return the chunk."""
    return str(rank) if rank is not None else "-"


def print_results(results: tuple[SearchResult, ...]) -> None:
    """One line per result with its fused score and per-leg ranks, then a snippet."""
    for position, result in enumerate(results, start=1):
        print(
            f"{position:>2}  {result.score:.4f}  {result.citation:<16} {result.celex}  "
            f"v{_rank(result.vector_rank)} t{_rank(result.text_rank)}"
        )
        print(f"    {result.text[:SNIPPET]}")


def print_article(chunks: tuple[RetrievedChunk, ...]) -> None:
    """The article in reading order, each chunk under its citation."""
    for chunk in chunks:
        print(f"{chunk.citation}  {chunk.celex}")
        print(f"    {chunk.text}")


async def _search(query: str, filters: SearchFilters, limit: int) -> tuple[SearchResult, ...]:
    async with get_session(auto_commit=False) as session:
        return await search(session, query, filters, limit=limit)


async def _article(celex: str, article: str) -> tuple[RetrievedChunk, ...]:
    async with get_session(auto_commit=False) as session:
        return await get_article(session, celex=celex, article=article)


def _run(args: argparse.Namespace) -> int:
    """Run the search or article branch and print its results; 0 on success, 1 on LLM failure."""
    try:
        if args.article:
            chunks = asyncio.run(_article(*args.article))
            print_article(chunks)
            found = bool(chunks)
        else:
            filters = SearchFilters(celex=args.celex, topic=args.topic)
            results = asyncio.run(_search(args.query, filters, args.limit))
            print_results(results)
            found = bool(results)
    except LLMError as exc:
        print(f"retrieval failed: {exc}", file=sys.stderr)
        return 1
    if not found:
        print("no results")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.query and not args.article:
        parser.error("give a query, or --article CELEX ARTICLE")
    setup_logging()
    return _run(args)
