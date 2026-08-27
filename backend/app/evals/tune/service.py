"""Run the grid: the golden dataset through retrieval alone, one point at a time."""

import logging
import time
from collections.abc import Sequence

from app.chat.graph import retrieve
from app.chat.models import ChatState
from app.core.clock import elapsed_ms
from app.core.config import config
from app.core.exceptions import DomainError
from app.evals.models import EvalCase, EvalDataset, EvalResult, RunSettings
from app.evals.service import select_cases
from app.evals.tune.metrics import compute_retrieval_metrics
from app.evals.tune.models import GridPoint, TunedPoint, TuneRun
from app.evals.tune.params import TUNABLE_PARAMS

logger = logging.getLogger(__name__)


async def run_retrieval(case: EvalCase) -> EvalResult:
    """One case through the retrieve node alone: what search found and what reached the
    prompt, with no model call to pay for. A case that raises is recorded, not raised."""
    state = ChatState(question=case.question)
    start = time.perf_counter()
    try:
        state.refresh(await retrieve(state) | {"question": case.question})
    except Exception as exc:
        state.record_error(exc)
        if isinstance(exc, DomainError):
            logger.warning("tune case %s failed: %s", case.id, state.error)
        else:
            logger.exception("tune case %s failed unexpectedly", case.id)
    state.total_ms = elapsed_ms(start)
    return EvalResult(case=case, state=state)


async def _measure_point(point: GridPoint, dataset: EvalDataset, pattern: str | None) -> TunedPoint:
    """The dataset run under the config as it now stands, recorded against its point."""
    results = [await run_retrieval(case) for case in select_cases(dataset, pattern)]
    return TunedPoint(point=point, metrics=compute_retrieval_metrics(results))


async def run_grid(
    dataset: EvalDataset, points: Sequence[GridPoint], pattern: str | None = None
) -> TuneRun:
    """Baseline first, then every point, the config restored to baseline between points
    — one factor's overrides must not leak into the next point's run."""
    baseline_values = {name: getattr(config, name) for name in TUNABLE_PARAMS}
    settings = RunSettings.from_config()
    results = [await _measure_point(GridPoint(), dataset, pattern)]
    for point in points:
        for name, value in point.overrides.items():
            setattr(config, name, value)
        results.append(await _measure_point(point, dataset, pattern))
        for name in point.overrides:
            setattr(config, name, baseline_values[name])
    return TuneRun(
        dataset_sha=dataset.sha256,
        case_pattern=pattern,
        settings=settings,
        results=tuple(results),
    )
