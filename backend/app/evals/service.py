"""Eval checks and runs against the corpus."""

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ChatNode
from app.chat.graph import chat_graph
from app.chat.models import ChatState
from app.core.clock import elapsed_ms, utc_now
from app.core.exceptions import DomainError
from app.evals.models import CaseResult, EvalCase, EvalDataset, RunResult, UnresolvedReference
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


def _failed_case(case: EvalCase, exc: Exception, start: float) -> CaseResult:
    """A case the graph raised on, logged where the traceback still exists and recorded
    with the exception's own type and detail rather than the HTTP summary."""
    if isinstance(exc, DomainError):
        logger.warning("eval case %s failed: %s", case.id, exc.message)
        detail = exc.message
    else:
        logger.exception("eval case %s failed unexpectedly", case.id)
        detail = f"{type(exc).__name__}: {exc}"
    return CaseResult(case=case, total_ms=elapsed_ms(start), error=detail)


async def run_case(case: EvalCase) -> CaseResult:
    """One case driven through the chat graph, timed as each node's update lands, and
    scored inside the try so a case that cannot be built costs its own row."""
    start = time.perf_counter()
    state: dict = {}
    retrieve_ms = 0
    try:
        async for update in chat_graph.astream(
            ChatState(question=case.question), stream_mode="updates"
        ):
            for node, payload in update.items():
                if node == ChatNode.RETRIEVE:
                    retrieve_ms = elapsed_ms(start)
                state |= payload
        usage = state.get("usage") or {}
        return CaseResult(
            case=case,
            hits=state.get("hits", ()),
            sources=state.get("sources", ()),
            answer=state.get("answer", ""),
            retrieve_ms=retrieve_ms,
            total_ms=elapsed_ms(start),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
    except Exception as exc:
        return _failed_case(case, exc, start)


async def run_dataset(dataset: EvalDataset, pattern: str | None = None) -> RunResult:
    """Every matching case, one at a time, so a per-case timing measures the case alone,
    stamped before the first rather than after the last."""
    started_at = utc_now()
    results = [await run_case(case) for case in select_cases(dataset, pattern)]
    return RunResult.from_results(results, dataset.sha256, started_at, pattern)
