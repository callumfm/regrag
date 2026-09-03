"""The golden dataset: what a case must carry, how the file loads and saves, and what
the dataset hash covers."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.evals.dataset.enums import EvalKind, EvalTrait
from app.evals.dataset.exceptions import EmptyDatasetError
from app.evals.dataset.models import (
    CaseReference,
    CaseSelection,
    CorpusStamp,
    EvalCase,
    EvalDataset,
)
from app.evals.dataset.stamp import save_dataset
from tests.evals.conftest import REFERENCE, eval_case, eval_dataset, out_of_corpus_case

STAMPED = CaseReference(celex="32023R1805", article="4", content_hashes=("a" * 12, "b" * 12))


def test_an_in_corpus_case_needs_an_answer_and_references() -> None:
    with pytest.raises(ValidationError, match="in_corpus"):
        eval_case(references=())
    with pytest.raises(ValidationError, match="in_corpus"):
        eval_case(answer=None)


def test_an_out_of_corpus_case_carries_neither() -> None:
    with pytest.raises(ValidationError, match="out_of_corpus"):
        EvalCase(id="x", kind=EvalKind.OUT_OF_CORPUS, question="q?", references=(REFERENCE,))
    with pytest.raises(ValidationError, match="out_of_corpus"):
        EvalCase(id="x", kind=EvalKind.OUT_OF_CORPUS, question="q?", answer="a")


def test_a_case_carries_no_traits_unless_marked() -> None:
    assert eval_case().traits == ()
    assert eval_case(traits=("multi_hop", "multi_part")).traits == (
        EvalTrait.MULTI_HOP,
        EvalTrait.MULTI_PART,
    )


def test_a_dataset_refuses_duplicate_case_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate case ids: x"):
        eval_dataset(eval_case(id="x"), eval_case(id="x"))


def test_the_golden_file_validates() -> None:
    assert EvalDataset.load().cases


# What the dataset hash covers


def test_the_hash_follows_the_cases_not_the_file() -> None:
    same = eval_dataset(eval_case()).sha256

    assert same == eval_dataset(eval_case()).sha256
    assert same != eval_dataset(eval_case(answer="b")).sha256


def test_the_hash_ignores_the_stamps() -> None:
    """A re-stamp records which text an answer was read against and changes nothing a run
    scores, so it must leave past runs of the same cases comparable."""
    unstamped = eval_dataset(eval_case(references=(REFERENCE,)))
    stamped = eval_dataset(eval_case(references=(STAMPED,)))

    assert unstamped.sha256 == stamped.sha256


def test_the_hash_ignores_the_traits() -> None:
    """A trait says what a case tests, not what it asserts, so marking one must leave past
    runs of the same cases comparable."""
    plain = eval_dataset(eval_case())
    marked = eval_dataset(eval_case(traits=(EvalTrait.MULTI_PART,)))

    assert plain.sha256 == marked.sha256


def test_the_hash_ignores_the_corpus_stamp() -> None:
    stamped = eval_dataset(eval_case()).model_copy(
        update={"corpus": CorpusStamp(corpus_version="2026-08-15-2cc038d", stamped_at="2026-08-28")}
    )

    assert stamped.sha256 == eval_dataset(eval_case()).sha256


def test_the_hash_still_follows_which_division_a_case_cites() -> None:
    """Only the stamp records the corpus; the target itself is part of what the case asserts."""
    elsewhere = CaseReference(celex="32023R1805", article="9")

    assert (
        eval_dataset(eval_case()).sha256 != eval_dataset(eval_case(references=(elsewhere,))).sha256
    )


# Loading, filtering and saving the dataset file


def _write_dataset(path: Path, *cases: EvalCase) -> Path:
    save_dataset(EvalDataset(cases=cases), path)
    return path


def test_load_filters_cases_by_id_and_records_the_filter(tmp_path: Path) -> None:
    file = _write_dataset(
        tmp_path / "golden.json", eval_case(id="fueleu-one"), eval_case(id="mrv-one")
    )

    dataset = EvalDataset.load(file, CaseSelection(id_contains="fueleu"))

    assert [case.id for case in dataset.cases] == ["fueleu-one", "mrv-one"]
    assert [case.id for case in dataset.selected_cases] == ["fueleu-one"]
    assert dataset.selection.id_contains == "fueleu"


def test_load_names_a_selection_that_matches_nothing(tmp_path: Path) -> None:
    file = _write_dataset(tmp_path / "golden.json", eval_case(id="fueleu-one"))

    with pytest.raises(EmptyDatasetError, match="nothing-here"):
        EvalDataset.load(file, CaseSelection(id_contains="nothing-here"))
    with pytest.raises(EmptyDatasetError, match="multi_hop"):
        EvalDataset.load(file, CaseSelection(trait=EvalTrait.MULTI_HOP))


def test_a_selection_narrows_on_trait_and_kind_as_well_as_id() -> None:
    split = eval_case(id="fueleu-split", traits=(EvalTrait.MULTI_PART,))
    both = eval_case(id="mrv-both", traits=(EvalTrait.MULTI_HOP, EvalTrait.MULTI_PART))
    dataset = eval_dataset(eval_case(id="fueleu-plain"), split, both, out_of_corpus_case())

    def selected(**criteria: Any) -> list[str]:
        chosen = dataset.model_copy(update={"selection": CaseSelection(**criteria)})
        return [case.id for case in chosen.selected_cases]

    assert selected() == ["fueleu-plain", "fueleu-split", "mrv-both", "ooc"]
    assert selected(trait=EvalTrait.MULTI_PART) == ["fueleu-split", "mrv-both"]
    assert selected(kind=EvalKind.OUT_OF_CORPUS) == ["ooc"]
    assert selected(id_contains="fueleu", trait=EvalTrait.MULTI_PART) == ["fueleu-split"]


def test_a_selection_says_whether_it_narrows_anything_and_how() -> None:
    assert not CaseSelection().selects_a_subset
    assert CaseSelection().describe() == "every case"

    narrowed = CaseSelection(id_contains="fueleu", trait=EvalTrait.MULTI_HOP)
    assert narrowed.selects_a_subset
    assert narrowed.describe() == "id_contains=fueleu trait=multi_hop"


def test_load_refuses_a_dataset_with_no_cases(tmp_path: Path) -> None:
    file = _write_dataset(tmp_path / "golden.json")

    with pytest.raises(EmptyDatasetError, match="no cases"):
        EvalDataset.load(file)


def test_the_hash_names_the_file_not_the_subset_scored(tmp_path: Path) -> None:
    """The hash compares a filtered spot-check against a full run of the same file."""
    file = _write_dataset(
        tmp_path / "golden.json", eval_case(id="fueleu-one"), eval_case(id="mrv-one")
    )

    subset = EvalDataset.load(file, CaseSelection(id_contains="fueleu"))

    assert subset.sha256 == EvalDataset.load(file).sha256
