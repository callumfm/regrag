"""Eval test factories shared across the eval test modules."""

from typing import Any

from app.chat.enums import ChatNode
from app.evals.enums import EvalKind
from app.evals.models import EvalCase, EvalDataset
from app.evals.results import CaseResult
from app.retrieval.models import ReferenceTarget
from tests.conftest import retrieved_chunk, search_result

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


def case_result(case: EvalCase | None = None, **overrides: Any) -> CaseResult:
    """A completed case result: gold retrieved, cited, and answered, overridable per field."""
    case = case or eval_case()
    defaults: dict[str, Any] = {
        "case": case,
        "nodes": (ChatNode.RETRIEVE, ChatNode.SYNTHESIZE),
        "hits": (search_result(),),
        "sources": (retrieved_chunk(),),
        "answer": "The limit applies [1].",
        "retrieve_ms": 100,
        "total_ms": 1000,
        "input_tokens": 1200,
        "output_tokens": 180,
    }
    return CaseResult(**{**defaults, **overrides})
