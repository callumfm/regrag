"""Evals CLI: `uv run evals check`, `uv run evals run [--case PATTERN] [--verbose]`."""

import argparse
import asyncio
from collections.abc import Sequence

from app.core.db.session import get_session
from app.core.logger import setup_logging
from app.evals.cache import enable_call_cache
from app.evals.metrics import format_rate, score_reference_citation_rate, score_reference_recall
from app.evals.models import EmptyError, EvalDataset, EvalResult, UnresolvedReference
from app.evals.service import evaluate_all_cases, find_unresolved_references
from app.evals.tune.cli import register_tune_command, run_tune


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="RegRag evals")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="confirm every case reference still resolves in the corpus")
    run = commands.add_parser("run", help="score the dataset against the current chat graph")
    run.add_argument("--case", help="only cases whose id contains this")
    run.add_argument("--verbose", action="store_true", help="list every case with its own scores")
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="pay for every embed and rerank again instead of replaying the cached ones",
    )
    register_tune_command(commands)
    return parser


def _format_case_line(result: EvalResult, width: int) -> str:
    """One case on one line: what search found, what reached the prompt, what the answer
    cited of the references the case authors, then how the run ended."""
    state, references = result.state, result.case.references
    scored = state.error is None
    recalled = scored and bool(references)
    raw = score_reference_recall(references, state.hits) if recalled else None
    expanded = score_reference_recall(references, state.sources) if recalled else None
    cited = (
        score_reference_citation_rate(state.answer, state.sources, references) if scored else None
    )
    return (
        f"{result.case.id:<{width}}  raw {format_rate(raw):>4}  exp {format_rate(expanded):>4}  "
        f"cite {format_rate(cited):>4}  {state.outcome.value:<8}{state.total_ms or 0:>6}ms"
        f"{'  ' + state.error if state.error else ''}"
    )


def format_case_lines(results: Sequence[EvalResult]) -> list[str]:
    """Every case as its own line, the id column sized to the longest id in the run. A case
    that raised scores nothing, as the aggregate leaves it out; one authoring no reference
    has no recall to measure, and prints a dash rather than a zero."""
    if not results:
        return []
    width = max(len(result.case.id) for result in results)
    return [_format_case_line(result, width) for result in results]


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


def run_evals(case_filter: str | None, verbose: bool = False, cached: bool = True) -> int:
    """Score the dataset and print what it measured, the cases first when asked for."""
    dataset = EvalDataset.load(case_filter=case_filter)

    if cached:
        enable_call_cache()

    run = asyncio.run(evaluate_all_cases(dataset))
    if verbose:
        print("\n".join(format_case_lines(run.results)), end="\n\n")

    print(run.summary())
    return 1 if run.metrics.errors else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    try:
        if args.command == "run":
            return run_evals(args.case, args.verbose, cached=not args.no_cache)

        if args.command == "tune":
            return run_tune(args.case, cached=not args.no_cache)
    except EmptyError as exc:
        print(exc)
        return 1

    return check_references()
