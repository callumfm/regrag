"""Eval values: one authored case of the golden dataset."""

from pydantic import model_validator

from app.core.models import FrozenModel
from app.evals.enums import EvalKind
from app.retrieval.models import ReferenceTarget


class EvalCase(FrozenModel):
    """A question, what a right answer must say, and where in the corpus it comes from."""

    id: str
    kind: EvalKind
    question: str
    reference_answer: str | None = None
    gold: tuple[ReferenceTarget, ...] = ()

    @model_validator(mode="after")
    def _kind_matches_fields(self) -> "EvalCase":
        """An in-corpus case is scored against gold and a reference; an out-of-corpus case
        is scored on refusal alone, so carrying either would be a mislabelled case."""
        has_evidence = bool(self.gold) and self.reference_answer is not None
        if self.kind is EvalKind.IN_CORPUS and not has_evidence:
            raise ValueError(f"{self.id}: an in_corpus case needs gold and a reference answer")
        if self.kind is EvalKind.OUT_OF_CORPUS and (self.gold or self.reference_answer):
            raise ValueError(f"{self.id}: an out_of_corpus case has neither gold nor reference")
        return self
