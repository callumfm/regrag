"""Tune values: a configuration under test, what a retrieval-only run of it measures,
and the whole tune run."""

from pydantic import Field, JsonValue

from app.core.models import FrozenModel
from app.evals.models import RunSettings


class GridPoint(FrozenModel):
    """One configuration under test: the settings it changes from baseline."""

    overrides: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def label(self) -> str:
        """The point as the table names it: its changed settings, or the baseline."""
        if not self.overrides:
            return "(baseline)"
        return " ".join(f"{name}={value}" for name, value in self.overrides.items())


class RetrievalMetrics(FrozenModel):
    """What a retrieval-only run measured. A rate is None when no case measured it.

    The retrieval subset of EvalMetrics — nothing a model call would fill — plus what
    the recall is bought with: mean context chunks and chars, each averaged over the
    scored in-corpus cases, and mean retrieve time, averaged over all scored cases —
    retrieve runs on refusals too.
    """

    cases: int
    in_corpus: int
    out_of_corpus: int
    errors: int
    raw_hit_rate: float | None
    raw_recall: float | None
    expanded_hit_rate: float | None
    expanded_recall: float | None
    gate_refusal_rate: float | None
    false_refusals: int
    refused_a_found_reference: int
    mean_context_chunks: float | None
    mean_context_chars: float | None
    mean_retrieve_ms: int


class TunedPoint(FrozenModel):
    """One grid point and what running the dataset under it measured."""

    point: GridPoint
    metrics: RetrievalMetrics


class TuneRun(FrozenModel):
    """One tune run: which dataset it scored, the baseline settings as provenance, and
    every point measured — the baseline itself always first."""

    dataset_sha: str
    case_pattern: str | None = None
    settings: RunSettings
    results: tuple[TunedPoint, ...]

    @property
    def baseline(self) -> TunedPoint:
        """The unmodified configuration every delta is read against."""
        return self.results[0]
