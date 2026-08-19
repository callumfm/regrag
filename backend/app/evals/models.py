"""Eval values: the golden dataset and the cases it holds."""

import hashlib
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter, model_validator

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
