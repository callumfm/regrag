"""Build the grid: which params vary, at which values, folded into points."""

from collections import Counter
from itertools import product
from typing import Any

from app.core.config import config
from app.evals.tune.models import GridPoint
from app.evals.tune.params import get_tunable_params, validate_value


def _parse_one(argument: str) -> tuple[str, list[Any]]:
    """One NAME=v1,v2 argument, each value validated before any run."""
    name, separator, raw = argument.partition("=")
    if not separator or not raw:
        raise ValueError(f"--set takes NAME=value[,value...], got {argument!r}")
    return name, [validate_value(name, value) for value in raw.split(",")]


def parse_param_values(arguments: list[str]) -> dict[str, list[Any]]:
    """Every --set argument parsed and validated, refusing a param named twice;
    no arguments at all means the full curated sweep."""
    if not arguments:
        return {name: list(values) for name, values in get_tunable_params().items()}
    parsed = [_parse_one(argument) for argument in arguments]
    duplicates = sorted(
        name for name, seen in Counter(name for name, _ in parsed).items() if seen > 1
    )
    if duplicates:
        raise ValueError(f"--set names a param twice: {', '.join(duplicates)}")
    return dict(parsed)


def build_points(values: dict[str, list[Any]], *, cross: bool) -> tuple[GridPoint, ...]:
    """The grid: each param varied alone against baseline, or the full product under
    --cross. A value equal to baseline adds nothing over the baseline run, so it is
    dropped from a point's overrides, and a point left with none is dropped whole."""
    deduped = {name: list(dict.fromkeys(candidates)) for name, candidates in values.items()}
    if not cross:
        return tuple(
            GridPoint(overrides={name: value})
            for name, candidates in deduped.items()
            for value in candidates
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
