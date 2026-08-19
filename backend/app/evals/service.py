"""Eval checks and runs against the corpus."""

import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ChatNode
from app.chat.graph import chat_graph
from app.chat.models import ChatState
from app.core.clock import elapsed_ms
from app.core.exceptions import describe
from app.evals.models import CaseResult, EvalCase, EvalDataset, RunResult, UnresolvedReference
from app.retrieval.follow import reference_exists


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


async def run_case(case: EvalCase) -> CaseResult:
    """One case driven through the chat graph, timed as each node's update lands.

    The graph is driven directly rather than through the SSE stream, so a run scores the
    answer without writing a chat_requests row for every case.
    """
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
    except Exception as exc:
        return CaseResult(case=case, total_ms=elapsed_ms(start), error=describe(exc)[1])

    total_ms = elapsed_ms(start)
    usage = state.get("usage")
    return CaseResult(
        case=case,
        hits=state.get("hits", ()),
        sources=state.get("sources", ()),
        answer=state.get("answer", ""),
        retrieve_ms=retrieve_ms,
        synthesize_ms=total_ms - retrieve_ms,
        total_ms=total_ms,
        input_tokens=usage["input_tokens"] if usage else None,
        output_tokens=usage["output_tokens"] if usage else None,
    )


async def run_dataset(dataset: EvalDataset, pattern: str | None = None) -> RunResult:
    """Every matching case, one at a time, so a per-case timing measures the case alone."""
    results = [await run_case(case) for case in select_cases(dataset, pattern)]
    return RunResult.from_results(results, dataset.sha256)
