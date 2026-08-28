"""Evals CLI: `uv run evals check | stamp | run [--case PATTERN] [--verbose] | tune`."""

import argparse
import asyncio
from collections.abc import Sequence

from app.core.db.session import get_session
from app.core.logger import setup_logging
from app.evals.cache import enable_call_cache
from app.evals.dataset.cli import check_dataset, register_dataset_commands, stamp_cases
from app.evals.dataset.models import DatasetDrift, EmptyError, EvalDataset
from app.evals.dataset.service import inspect_dataset
from app.evals.metrics import format_rate, score_reference_citation_rate, score_reference_recall
from app.evals.models import EvalResult
from app.evals.service import evaluate_all_cases
from app.evals.tune.cli import register_tune_command, run_tune
from app.ingestion.service import get_latest_corpus_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="RegRag evals")
    commands = parser.add_subparsers(dest="command", required=True)
    register_dataset_commands(commands)
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


async def _read_provenance(dataset: EvalDataset) -> tuple[str | None, DatasetDrift]:
    """What the corpus stands at and how far the dataset has drifted from it, read once
    before the run so a score carries the state of the text it was measured against."""
    async with get_session(auto_commit=False) as session:
        return await get_latest_corpus_version(session), await inspect_dataset(session, dataset)


def run_evals(case_filter: str | None, verbose: bool = False, cached: bool = True) -> int:
    """Score the dataset and print what it measured, the cases first when asked for."""
    dataset = EvalDataset.load(case_filter=case_filter)
    corpus_version, drift = asyncio.run(_read_provenance(dataset))

    if cached:
        enable_call_cache()

    run = asyncio.run(evaluate_all_cases(dataset, corpus_version, drift.stale_case_ids))
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

        if args.command == "stamp":
            return stamp_cases(args.case)

        return check_dataset()
    except EmptyError as exc:
        print(exc)
        return 1
