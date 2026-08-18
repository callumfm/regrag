"""Eval test factories shared across the eval test modules."""

from typing import Any

from app.evals.enums import EvalKind
from app.evals.models import EvalCase, EvalDataset
from app.retrieval.models import ReferenceTarget

REFERENCE = ReferenceTarget(celex="32023R1805", article="4")

IN_CORPUS_CASE: dict[str, Any] = {
    "id": "case",
    "kind": EvalKind.IN_CORPUS,
    "question": "q?",
    "answer": "a",
    "references": (REFERENCE,),
}


def eval_case(**overrides: Any) -> EvalCase:
    """An in-corpus case with sane defaults, overridable per field."""
    return EvalCase(**{**IN_CORPUS_CASE, **overrides})


def out_of_corpus_case(id: str = "ooc") -> EvalCase:
    return EvalCase(id=id, kind=EvalKind.OUT_OF_CORPUS, question="q?")


def eval_dataset(*cases: EvalCase) -> EvalDataset:
    return EvalDataset(cases=cases)
