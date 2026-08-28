"""Eval values: the golden dataset and its cases, and what a run of it produces."""

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, model_validator

from app.chat.models import ChatState
from app.core.config import config
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


class EmptyError(ValueError):
    """A dataset load that selected no cases."""


class EvalDataset(FrozenModel):
    """The golden dataset: every authored case, and the filter naming the subset a run scores."""

    cases: tuple[EvalCase, ...]
    case_filter: str | None = None

    @classmethod
    def load(
        cls, path: Path = config.EVAL_DATASET_PATH, case_filter: str | None = None
    ) -> "EvalDataset":
        """Read and validate the JSON file at path."""
        cases = _cases.validate_json(path.read_bytes())
        if not cases:
            raise EmptyError("The dataset has no cases")

        dataset = cls(cases=cases, case_filter=case_filter)
        if not dataset.selected_cases:
            raise EmptyError(f"No cases found matching filter: {case_filter}")
        return dataset

    @property
    def selected_cases(self) -> tuple[EvalCase, ...]:
        """The cases a run scores: those the filter matches, or every case without one."""
        if self.case_filter is None:
            return self.cases
        return tuple(case for case in self.cases if self.case_filter in case.id)

    @property
    def sha256(self) -> str:
        """Hash of every authored case as canonical JSON — the whole dataset however a run
        filters it, so a filtered spot-check still names the file a full run scored."""
        return hashlib.sha256(self.model_dump_json(include={"cases"}).encode()).hexdigest()

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


class RetrievalMetrics(FrozenModel):
    """The measures retrieval alone fills. A rate is None when no case measured it.

    in_corpus / out_of_corpus: how the run's cases were authored, counted whether or not
        they scored, so errors overlaps them rather than partitioning with them.
    raw_*: scored on what search found; expanded_*: on what reached the prompt.
    gate_refusal_rate: out-of-corpus cases the pre-model gate refused; a model declining in
        its own words is the judge's to score.
    false_refusals: in-corpus cases the gate refused; refused_a_found_reference: those where
        search had already found an authored reference — the gate too tight.
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


class EvalMetrics(RetrievalMetrics):
    """The retrieval measures plus what the model calls added.

    cited_references: share of authored references the answers cited.
    markers_in_context: share of [n] markers addressing a block that was in context.
    mean_node_ms: a node's mean time over the cases that ran it.
    """

    cited_references: float | None
    markers_in_context: float | None
    mean_node_ms: dict[str, int]
    mean_total_ms: int
    input_tokens: int
    output_tokens: int


class EvalRun(FrozenModel):
    """One eval run: which dataset and settings it scored, what it measured, and every case.

    dataset_sha hashes the whole dataset; case_filter names the subset actually scored.
    cached says the run had the call cache on, so an embed or rerank timing may measure a
    disk read rather than the provider — a cached run is not a latency baseline.
    """

    dataset_sha: str
    case_filter: str | None = None
    cached: bool = False
    settings: dict[str, Any]
    metrics: EvalMetrics
    results: tuple[EvalResult, ...]

    def summary(self) -> str:
        """The provenance and scores as JSON, then any case the graph raised on."""
        body = self.model_dump_json(
            indent=2, include={"dataset_sha", "case_filter", "cached", "settings", "metrics"}
        )
        errored = [f"  {r.case.id}  {r.state.error}" for r in self.results if r.state.error]
        return "\n".join([body, "", "errored:", *errored]) if errored else body
