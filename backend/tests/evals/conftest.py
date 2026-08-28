"""Eval test factories shared across the eval test modules."""

from typing import Any

import pytest

from app.chat.enums import ChatNode
from app.chat.models import ChatNodeResult, ChatState
from app.chat.prompts import REFUSAL_ANSWER
from app.core.config import config
from app.evals.dataset.enums import EvalKind
from app.evals.dataset.models import CaseReference, EvalCase, EvalDataset
from app.evals.models import EvalResult
from tests.conftest import retrieved_chunk, search_result

REFERENCE = CaseReference(celex="32023R1805", article="4")


@pytest.fixture(autouse=True)
def no_assess_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval tests fake the chat model, not assess — the loop stays off here; its
    coverage lives in tests/chat."""
    monkeypatch.setattr(config, "ASSESS_ENABLED", False)


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


def eval_dataset(*cases: EvalCase, case_filter: str | None = None) -> EvalDataset:
    return EvalDataset(cases=cases, case_filter=case_filter)


def eval_result(case: EvalCase | None = None, **state: Any) -> EvalResult:
    """A completed in-corpus case whose answer cites its one authored reference, with the
    state's fields overridable — nodes, hits, sources, answer, error."""
    defaults: dict[str, Any] = {
        "question": "q?",
        "nodes": (
            ChatNodeResult(node=ChatNode.RETRIEVE, ms=100),
            ChatNodeResult(node=ChatNode.SYNTHESIZE, ms=900, input_tokens=1500, output_tokens=40),
        ),
        "hits": (search_result(),),
        "sources": (retrieved_chunk(),),
        "answer": "Yes [1].",
        "total_ms": 1000,
    }
    return EvalResult(case=case or eval_case(), state=ChatState(**{**defaults, **state}))


REFUSED_PATH = (
    ChatNodeResult(node=ChatNode.RETRIEVE, ms=80),
    ChatNodeResult(node=ChatNode.REFUSE, ms=0),
)
"""The path a gate refusal leaves: retrieve ran, then refuse, and no model call."""


def refused_result(case: EvalCase | None = None, **state: Any) -> EvalResult:
    """A case the gate refused: the refusal path, no sources, the fixed answer."""
    defaults: dict[str, Any] = {
        "nodes": REFUSED_PATH,
        "hits": (),
        "sources": (),
        "answer": REFUSAL_ANSWER,
        "total_ms": 85,
    }
    return EvalResult(
        case=case or out_of_corpus_case(), state=ChatState(question="q?", **{**defaults, **state})
    )
