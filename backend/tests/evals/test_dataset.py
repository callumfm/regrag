import pytest
from pydantic import ValidationError

from app.evals.dataset import dataset_hash, load_golden
from app.evals.enums import EvalKind
from app.evals.models import EvalCase
from app.retrieval.models import ReferenceTarget

GOLD = (ReferenceTarget(celex="32023R1805", article="23"),)


def test_an_in_corpus_case_needs_gold_and_a_reference_answer() -> None:
    with pytest.raises(ValidationError, match="in_corpus"):
        EvalCase(id="x", kind=EvalKind.IN_CORPUS, question="q?", reference_answer="a")
    with pytest.raises(ValidationError, match="in_corpus"):
        EvalCase(id="x", kind=EvalKind.IN_CORPUS, question="q?", gold=GOLD)


def test_an_out_of_corpus_case_carries_neither() -> None:
    with pytest.raises(ValidationError, match="out_of_corpus"):
        EvalCase(id="x", kind=EvalKind.OUT_OF_CORPUS, question="q?", gold=GOLD)
    with pytest.raises(ValidationError, match="out_of_corpus"):
        EvalCase(id="x", kind=EvalKind.OUT_OF_CORPUS, question="q?", reference_answer="a")


def test_the_golden_file_loads_with_unique_ids() -> None:
    cases = load_golden()

    assert cases
    assert len({case.id for case in cases}) == len(cases)


def test_the_dataset_hash_is_stable() -> None:
    assert dataset_hash() == dataset_hash()
    assert len(dataset_hash()) == 64
