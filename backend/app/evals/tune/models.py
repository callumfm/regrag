"""Tune values: a configuration under test, what a retrieval-only run of it measures,
and the whole tune run."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import Field

from app.core.config import Config, config
from app.core.models import FrozenModel
from app.evals.models import RetrievalMetrics


class TunableParam(FrozenModel):
    """A config parameter and the values to test for it."""

    name: str
    values: tuple[Any, ...]

    def validate_config(self) -> None:
        if self.name not in Config.model_fields:
            raise ValueError(f"{self.name} is no longer a config field; update TUNABLE_PARAMS")

        probe = config.model_copy()
        for value in self.values:
            setattr(probe, self.name, value)

    @contextmanager
    def override(self, value: Any) -> Iterator[None]:
        """Temporarily apply a value to this parameter."""
        previous = getattr(config, self.name)

        try:
            setattr(config, self.name, value)
            yield
        finally:
            setattr(config, self.name, previous)


class TuneMetrics(RetrievalMetrics):
    """Retrieval metrics plus context and retrieval cost."""

    mean_context_chunks: float | None
    mean_context_chars: float | None
    mean_retrieve_ms: int | None


class TuneResult(FrozenModel):
    """The result of trying one value for one config parameter."""

    param: str
    value: Any
    metrics: TuneMetrics


class TuneRun(FrozenModel):
    """A complete tuning run against a fixed baseline configuration."""

    dataset_sha: str
    case_filter: str | None = None
    cached: bool = False
    settings: dict[str, Any]
    baseline: TuneMetrics
    results: tuple[TuneResult, ...] = Field(default_factory=tuple)
