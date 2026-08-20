"""Eval checks and runs against the corpus."""

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.graph import chat_graph
from app.chat.models import ChatState
from app.core.clock import elapsed_ms
from app.core.exceptions import DomainError
from app.evals.cache import call_cache_enabled
from app.evals.metrics import compute_metrics
from app.evals.models import (
    EvalCase,
    EvalDataset,
    EvalResult,
    EvalRun,
    RunSettings,
    UnresolvedReference,
)
from app.retrieval.follow import reference_exists

logger = logging.getLogger(__name__)


async def find_unresolved_references(
    session: AsyncSession, dataset: EvalDataset
) -> tuple[UnresolvedReference, ...]:
    """Every case reference with no stored chunk for its celex + article/annex, with its case id.
    Stale after a renumbered re-ingest or a typo; a run would score it as a retrieval miss."""
    unresolved = []
    for case in dataset.cases:
        for target in case.references:
            if not await reference_exists(session, target):
                unresolved.append(UnresolvedReference(case_id=case.id, target=target))
    return tuple(unresolved)


def select_cases(dataset: EvalDataset, pattern: str | None = None) -> tuple[EvalCase, ...]:
    """The cases whose id contains the pattern, or all of them when none is given."""
    if pattern is None:
        return dataset.cases
    return tuple(case for case in dataset.cases if pattern in case.id)


async def run_case(case: EvalCase) -> EvalResult:
    """One case driven through the chat graph, ending in the state a chat request ends in.
    A case the graph raises on is recorded by name, not raised: the run goes on."""
    state = ChatState(question=case.question)
    start = time.perf_counter()
    try:
        state.refresh(await chat_graph.ainvoke(state))
    except Exception as exc:
        state.record_error(exc)
        if isinstance(exc, DomainError):
            logger.warning("eval case %s failed: %s", case.id, state.error)
        else:
            logger.exception("eval case %s failed unexpectedly", case.id)
    state.total_ms = elapsed_ms(start)
    return EvalResult(case=case, state=state)


async def run_dataset(dataset: EvalDataset, pattern: str | None = None) -> EvalRun:
    """Every matching case, one at a time, so a per-case timing measures the case alone."""
    results = [await run_case(case) for case in select_cases(dataset, pattern)]
    return EvalRun(
        dataset_sha=dataset.sha256,
        case_pattern=pattern,
        cached=call_cache_enabled(),
        settings=RunSettings.from_config(),
        metrics=compute_metrics(results),
        results=tuple(results),
    )
