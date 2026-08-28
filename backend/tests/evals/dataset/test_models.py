"""The golden dataset: what a case must carry, how the file loads and saves, and what
the dataset hash covers."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evals.dataset.enums import EvalKind
from app.evals.dataset.models import CaseReference, CorpusStamp, EmptyError, EvalCase, EvalDataset
from tests.evals.conftest import REFERENCE, eval_case, eval_dataset

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


def test_the_hash_ignores_the_corpus_stamp() -> None:
    stamped = eval_dataset(eval_case()).model_copy(
        update={"corpus": CorpusStamp(stamped_at="2026-08-28", documents={"32023R1805": "9f2c"})}
    )

    assert stamped.sha256 == eval_dataset(eval_case()).sha256


def test_the_hash_still_follows_which_division_a_case_cites() -> None:
    """Only the stamp is provenance; the target itself is part of what the case asserts."""
    elsewhere = CaseReference(celex="32023R1805", article="9")

    assert (
        eval_dataset(eval_case()).sha256 != eval_dataset(eval_case(references=(elsewhere,))).sha256
    )


# Loading, filtering and saving the dataset file


def _write_dataset(path: Path, *cases: EvalCase, corpus: CorpusStamp | None = None) -> Path:
    EvalDataset(cases=cases, corpus=corpus).save(path)
    return path


def test_load_filters_cases_by_id_and_records_the_filter(tmp_path: Path) -> None:
    file = _write_dataset(
        tmp_path / "golden.json", eval_case(id="fueleu-one"), eval_case(id="mrv-one")
    )

    dataset = EvalDataset.load(file, case_filter="fueleu")

    assert [case.id for case in dataset.cases] == ["fueleu-one", "mrv-one"]
    assert [case.id for case in dataset.selected_cases] == ["fueleu-one"]
    assert dataset.case_filter == "fueleu"


def test_load_names_a_filter_that_matches_nothing(tmp_path: Path) -> None:
    file = _write_dataset(tmp_path / "golden.json", eval_case(id="fueleu-one"))

    with pytest.raises(EmptyError, match="nothing-here"):
        EvalDataset.load(file, case_filter="nothing-here")


def test_load_refuses_a_dataset_with_no_cases(tmp_path: Path) -> None:
    file = _write_dataset(tmp_path / "golden.json")

    with pytest.raises(EmptyError, match="no cases"):
        EvalDataset.load(file)


def test_the_hash_names_the_file_not_the_subset_scored(tmp_path: Path) -> None:
    """Provenance compares a filtered spot-check against a full run of the same file."""
    file = _write_dataset(
        tmp_path / "golden.json", eval_case(id="fueleu-one"), eval_case(id="mrv-one")
    )

    assert EvalDataset.load(file, case_filter="fueleu").sha256 == EvalDataset.load(file).sha256


def test_a_saved_dataset_loads_back_as_the_same_cases_and_stamp(tmp_path: Path) -> None:
    corpus = CorpusStamp(
        corpus_version="2026-08-01-a3f1c2", stamped_at="2026-08-28", documents={"32023R1805": "9f"}
    )
    file = _write_dataset(
        tmp_path / "golden.json",
        eval_case(id="stamped", references=(STAMPED,)),
        EvalCase(id="ooc", kind=EvalKind.OUT_OF_CORPUS, question="q?"),
        corpus=corpus,
    )

    loaded = EvalDataset.load(file)

    assert loaded.corpus == corpus
    assert loaded.cases[0].references == (STAMPED,)
    assert loaded.cases[1].references == ()


def test_saving_keeps_a_reference_on_one_line_and_omits_what_it_does_not_carry(
    tmp_path: Path,
) -> None:
    """The file is hand-authored and PR-reviewed, so a re-stamp has to diff as the one line
    that moved rather than reformatting every case around it."""
    file = _write_dataset(tmp_path / "golden.json", eval_case(references=(STAMPED,)))

    text = file.read_text()

    assert f'{{"celex": "32023R1805", "article": "4", "content_hashes": ["{"a" * 12}", ' in text
    assert '"annex"' not in text
    assert '"case_filter"' not in text


def test_the_dataset_names_the_cases_no_stamp_was_ever_recorded_for() -> None:
    """An unstamped case is not stale — nothing was recorded to compare against — but it is
    a case drift cannot be seen on, so it is named rather than passed over in silence."""
    dataset = eval_dataset(
        eval_case(id="stamped", references=(STAMPED,)),
        eval_case(id="never-stamped", references=(REFERENCE,)),
        EvalCase(id="ooc", kind=EvalKind.OUT_OF_CORPUS, question="q?"),
    )

    assert dataset.unstamped_cases == ("never-stamped",)
