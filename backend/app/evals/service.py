"""Eval checks and runs against the corpus."""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import litellm
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.graph import chat_graph
from app.chat.models import ChatState
from app.core.clock import elapsed_ms
from app.core.config import EVAL_CONFIG_SECTIONS, get_config_snapshot
from app.core.exceptions import DomainError
from app.evals.metrics import compute_metrics
from app.evals.models import (
    EvalCase,
    EvalDataset,
    EvalResult,
    EvalRun,
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


EvalGraph = Callable[[ChatState], Awaitable[dict[str, Any]]]
"""Which set of nodes to use when evaluating an EvalCase."""


async def _full_chat_graph(state: ChatState) -> dict[str, Any]:
    return await chat_graph.ainvoke(state)


async def evaluate_case(case: EvalCase, graph: EvalGraph = _full_chat_graph) -> EvalResult:
    """One case driven to the state a chat request ends in — through the whole chat graph
    unless told otherwise. A case the driver raises on is recorded by name, not raised:
    the run goes on."""
    state = ChatState(question=case.question)
    start = time.perf_counter()
    try:
        state.refresh(await graph(state))
    except Exception as exc:
        state.record_error(exc)
        if isinstance(exc, DomainError):
            logger.warning("eval case %s failed: %s", case.id, state.error)
        else:
            logger.exception("eval case %s failed unexpectedly", case.id)
    state.total_ms = elapsed_ms(start)
    return EvalResult(case=case, state=state)


async def evaluate_all_cases(dataset: EvalDataset) -> EvalRun:
    """Every case in the dataset, one at a time, so a per-case timing measures the case alone.
    Whether the run was cached is read off the live litellm cache, not a caller's word,
    so the provenance cannot disagree with what served the calls."""
    results = [await evaluate_case(case) for case in dataset.cases]
    settings = get_config_snapshot(EVAL_CONFIG_SECTIONS)
    return EvalRun(
        dataset_sha=dataset.sha256,
        case_filter=dataset.case_filter,
        cached=litellm.cache is not None,
        settings=settings,
        metrics=compute_metrics(results),
        results=tuple(results),
    )
