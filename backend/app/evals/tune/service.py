"""Grid tuning: run the golden dataset across retrieval settings, one point at a time."""

import logging
import time
from collections import Counter
from collections.abc import Sequence
from itertools import product
from typing import Any

from pydantic import SecretStr

from app.chat.graph import retrieve
from app.chat.models import ChatState
from app.core.clock import elapsed_ms
from app.core.config import BaseConfig, ChatConfig, RetrievalConfig, config
from app.core.exceptions import DomainError
from app.evals.models import EvalCase, EvalDataset, EvalResult, RunSettings
from app.evals.service import select_cases
from app.evals.tune.metrics import compute_retrieval_metrics
from app.evals.tune.models import GridPoint, TunedPoint, TuneRun

logger = logging.getLogger(__name__)

_TUNABLE_SECTIONS = (ChatConfig, RetrievalConfig)
"""The sections a grid may vary: what the chat graph and its retrieval read, and what
RunSettings records — so nothing can be tuned that a run would not record."""


def tunable_fields() -> dict[str, type[BaseConfig]]:
    """Every setting tune may vary, mapped to the section that validates it."""
    return {
        name: section
        for section in _TUNABLE_SECTIONS
        for name, field in section.model_fields.items()
        if field.annotation is not SecretStr
    }


def _parse_one(argument: str, sections: dict[str, type[BaseConfig]]) -> tuple[str, list[Any]]:
    """One NAME=v1,v2 argument, each value validated by its section before any run."""
    name, separator, raw = argument.partition("=")
    if name not in sections:
        valid = ", ".join(sorted(sections))
        raise ValueError(f"{name} is not a tunable setting; expected one of: {valid}")
    if not separator or not raw:
        raise ValueError(f"--set takes NAME=value[,value...], got {argument!r}")
    return name, [
        getattr(sections[name].model_validate({name: value}), name) for value in raw.split(",")
    ]


def parse_settings(arguments: list[str]) -> dict[str, list[Any]]:
    """Every --set argument parsed and validated, refusing a setting named twice."""
    sections = tunable_fields()
    parsed = [_parse_one(argument, sections) for argument in arguments]
    duplicates = sorted(
        name for name, seen in Counter(name for name, _ in parsed).items() if seen > 1
    )
    if duplicates:
        raise ValueError(f"--set names a setting twice: {', '.join(duplicates)}")
    return dict(parsed)


def build_points(settings: dict[str, list[Any]], *, cross: bool) -> tuple[GridPoint, ...]:
    """The grid: each setting varied alone against baseline, or the full product under
    --cross. A value equal to baseline adds nothing over the baseline run, so it is
    dropped from a point's overrides, and a point left with none is dropped whole."""
    deduped = {name: list(dict.fromkeys(values)) for name, values in settings.items()}
    if not cross:
        return tuple(
            GridPoint(overrides={name: value})
            for name, values in deduped.items()
            for value in values
            if value != getattr(config, name)
        )
    points = []
    for combination in product(*deduped.values()):
        overrides = {
            name: value
            for name, value in zip(deduped, combination, strict=True)
            if value != getattr(config, name)
        }
        if overrides:
            points.append(GridPoint(overrides=overrides))
    return tuple(points)


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
    baseline_values = {name: getattr(config, name) for name in tunable_fields()}
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
