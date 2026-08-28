"""Driving the golden cases through the chat graph and recording what the run measured."""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import litellm

from app.chat.graph import chat_graph
from app.chat.models import ChatState
from app.core.clock import elapsed_ms
from app.core.config import EVAL_CONFIG_SECTIONS, get_config_snapshot
from app.core.exceptions import DomainError
from app.evals.dataset.models import EvalCase, EvalDataset
from app.evals.metrics import compute_metrics
from app.evals.models import EvalResult, EvalRun

logger = logging.getLogger(__name__)


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
        state.sync_from_snapshot(await graph(state))
    except Exception as exc:
        state.record_error(exc)
        if isinstance(exc, DomainError):
            logger.warning("eval case %s failed: %s", case.id, state.error)
        else:
            logger.exception("eval case %s failed unexpectedly", case.id)
    state.total_ms = elapsed_ms(start)
    return EvalResult(case=case, state=state)


async def evaluate_all_cases(
    dataset: EvalDataset,
    corpus_version: str | None = None,
    stale_cases: tuple[str, ...] = (),
) -> EvalRun:
    """Every case in the dataset, one at a time, so a per-case timing measures the case alone.

    The corpus version and the stale cases are read before the run and carried through it, so
    a score always says which text it was measured against and which cases owe a re-review.
    Whether the run was cached is read off the live litellm cache, not a caller's word, so the
    recorded flag cannot disagree with what served the calls.
    """
    results = [await evaluate_case(case) for case in dataset.selected_cases]
    settings = get_config_snapshot(EVAL_CONFIG_SECTIONS)
    return EvalRun(
        dataset_sha=dataset.sha256,
        case_filter=dataset.case_filter,
        corpus_version=corpus_version,
        stale_cases=stale_cases,
        cached=litellm.cache is not None,
        settings=settings,
        metrics=compute_metrics(results),
        results=tuple(results),
    )
