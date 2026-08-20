"""Eval values: the golden dataset and its cases, and what a run of it produces."""

import hashlib
from collections import Counter
from pathlib import Path

from pydantic import ConfigDict, JsonValue, RootModel, SecretStr, TypeAdapter, model_validator

from app.chat.models import ChatState
from app.core.config import ChatConfig, EmbeddingConfig, RetrievalConfig, config
from app.core.models import FrozenModel
from app.evals.enums import EvalKind
from app.retrieval.models import ReferenceTarget


class EvalCase(FrozenModel):
    """A question, what a right answer must say, and where in the corpus it comes from."""

    id: str
    kind: EvalKind
    question: str
    answer: str | None = None
    references: tuple[ReferenceTarget, ...] = ()

    @model_validator(mode="after")
    def _kind_matches_fields(self) -> "EvalCase":
        """An in-corpus case is scored against its answer and references; an out-of-corpus case
        is scored on refusal alone, so carrying either would be a mislabelled case."""
        has_evidence = bool(self.references) and self.answer is not None
        if self.kind is EvalKind.IN_CORPUS and not has_evidence:
            raise ValueError(f"{self.id}: an in_corpus case needs an answer and references")
        if self.kind is EvalKind.OUT_OF_CORPUS and (self.references or self.answer):
            raise ValueError(f"{self.id}: an out_of_corpus case has neither answer nor references")
        return self


_cases = TypeAdapter(tuple[EvalCase, ...])


class EvalDataset(FrozenModel):
    """The golden dataset: every authored case."""

    cases: tuple[EvalCase, ...]

    @classmethod
    def load(cls, path: Path = config.EVAL_DATASET_PATH) -> "EvalDataset":
        """Read and validate the JSON file at path."""
        return cls(cases=_cases.validate_json(path.read_bytes()))

    @property
    def sha256(self) -> str:
        """Hash of the cases as canonical JSON, so a run records exactly which dataset it scored."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "EvalDataset":
        """A case id is how a result names its case, so two cases cannot share one."""
        counts = Counter(case.id for case in self.cases)
        duplicates = sorted(id for id, seen in counts.items() if seen > 1)
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        return self


class UnresolvedReference(FrozenModel):
    """A case reference no stored chunk answers to."""

    case_id: str
    target: ReferenceTarget


class EvalResult(FrozenModel):
    """One case driven through the chat graph: the case, and the run it produced — the
    same state a chat request ends in, so a run is scored off what production records."""

    case: EvalCase
    state: ChatState


_RECORDED_SECTIONS = (EmbeddingConfig, ChatConfig, RetrievalConfig)
"""The settings sections a run reads; taken whole, so a new knob is recorded without
anyone remembering to add it here."""


class RunSettings(RootModel[dict[str, JsonValue]]):
    """Every setting the chat graph and its retrieval read, keyed by the environment
    variable that sets it, so two runs are only compared when comparable. A secret is
    typed SecretStr and is not a setting a run reproduces, so it is left out."""

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_config(cls) -> "RunSettings":
        """The settings as the run will really read them."""
        return cls(
            {
                name: getattr(config, name)
                for section in _RECORDED_SECTIONS
                for name, field in sorted(section.model_fields.items())
                if field.annotation is not SecretStr
            }
        )


class EvalMetrics(FrozenModel):
    """What a run measured. A rate is None when no case measured it — unmeasured, not zero.

    in_corpus / out_of_corpus: how the run's cases were authored, counted whether or not
        they scored, so errors overlaps them rather than partitioning with them.
    raw_*: scored on what search found; expanded_*: on what reached the prompt.
    cited_references: share of authored references the answers cited.
    markers_in_context: share of [n] markers addressing a block that was in context.
    gate_refusal_rate: out-of-corpus cases the pre-model gate refused; a model declining in
        its own words is the judge's to score.
    false_refusals: in-corpus cases the gate refused; refused_a_found_reference: those where
        search had already found an authored reference — the gate too tight.
    mean_node_ms: a node's mean time over the cases that ran it.
    """

    cases: int
    in_corpus: int
    out_of_corpus: int
    errors: int
    raw_hit_rate: float | None
    raw_recall: float | None
    expanded_hit_rate: float | None
    expanded_recall: float | None
    cited_references: float | None
    markers_in_context: float | None
    gate_refusal_rate: float | None
    false_refusals: int
    refused_a_found_reference: int
    mean_node_ms: dict[str, int]
    mean_total_ms: int
    input_tokens: int
    output_tokens: int


class EvalRun(FrozenModel):
    """One eval run: which dataset and settings it scored, what it measured, and every case.

    dataset_sha hashes the whole dataset; case_pattern names the subset actually scored.
    cached says the run had the call cache on, so an embed or rerank timing may measure a
    disk read rather than the provider — a cached run is not a latency baseline.
    """

    dataset_sha: str
    case_pattern: str | None = None
    cached: bool = False
    settings: RunSettings
    metrics: EvalMetrics
    results: tuple[EvalResult, ...]

    def summary(self) -> str:
        """The provenance and scores as JSON, then any case the graph raised on."""
        body = self.model_dump_json(
            indent=2, include={"dataset_sha", "case_pattern", "cached", "settings", "metrics"}
        )
        errored = [f"  {r.case.id}  {r.state.error}" for r in self.results if r.state.error]
        return "\n".join([body, "", "errored:", *errored]) if errored else body
