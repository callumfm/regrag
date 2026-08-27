"""Grid tuning: run the golden dataset across retrieval settings, one point at a time."""

from collections import Counter
from itertools import product
from typing import Any

from pydantic import SecretStr

from app.core.config import BaseConfig, ChatConfig, RetrievalConfig, config
from app.evals.tune.models import GridPoint

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
