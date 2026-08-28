"""Run retrieval tuning against the golden dataset, one parameter value at a time."""

from collections.abc import Sequence
from typing import Any

import litellm

from app.chat.graph import retrieve
from app.chat.models import ChatState
from app.core.config import EVAL_CONFIG_SECTIONS, config, get_config_snapshot
from app.evals.models import EvalCase, EvalDataset, EvalResult
from app.evals.service import evaluate_case
from app.evals.tune.metrics import compute_tune_metrics
from app.evals.tune.models import TunableParam, TuneMetrics, TuneResult, TuneRun


async def retrieve_graph(state: ChatState) -> dict[str, Any]:
    """Run retrieval only, carrying the question through the state update."""
    return await retrieve(state) | {"question": state.question}


async def _measure(cases: tuple[EvalCase, ...]) -> TuneMetrics:
    """Run the selected cases through retrieval and calculate tuning metrics."""
    results: list[EvalResult] = [await evaluate_case(case, graph=retrieve_graph) for case in cases]
    return compute_tune_metrics(results)


async def tune(dataset: EvalDataset, params: Sequence[TunableParam]) -> TuneRun:
    """Measure the baseline, then each parameter value independently."""
    for param in params:
        param.validate_config()

    settings = get_config_snapshot(EVAL_CONFIG_SECTIONS)
    baseline = await _measure(dataset.selected_cases)

    results: list[TuneResult] = []
    for param in params:
        for value in param.values:
            if value == getattr(config, param.name):
                continue

            with param.override(value):
                metrics = await _measure(dataset.selected_cases)

            result = TuneResult(
                param=param.name, value=value, requires=param.requires, metrics=metrics
            )
            results.append(result)

    return TuneRun(
        dataset_sha=dataset.sha256,
        case_filter=dataset.case_filter,
        cached=litellm.cache is not None,
        settings=settings,
        baseline=baseline,
        results=tuple(results),
    )
