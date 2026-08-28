import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import EVAL_CONFIG_SECTIONS, config, get_config_snapshot
from app.evals.enums import EvalKind
from app.evals.metrics import compute_metrics
from app.evals.models import EmptyError, EvalCase, EvalDataset, EvalRun
from tests.evals.conftest import REFERENCE, eval_case, eval_dataset, eval_result


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


def test_the_hash_follows_the_cases_not_the_file() -> None:
    same = eval_dataset(eval_case()).sha256

    assert same == eval_dataset(eval_case()).sha256
    assert same != eval_dataset(eval_case(answer="b")).sha256


# Loading and filtering the dataset file


def _write_dataset(path: Path, *cases: EvalCase) -> Path:
    path.write_text(json.dumps([case.model_dump(mode="json") for case in cases]))
    return path


def test_load_filters_cases_by_id_and_records_the_filter(tmp_path: Path) -> None:
    file = _write_dataset(
        tmp_path / "golden.json", eval_case(id="fueleu-one"), eval_case(id="mrv-one")
    )

    dataset = EvalDataset.load(file, case_filter="fueleu")

    assert [case.id for case in dataset.cases] == ["fueleu-one"]
    assert dataset.case_filter == "fueleu"


def test_load_names_a_filter_that_matches_nothing(tmp_path: Path) -> None:
    file = _write_dataset(tmp_path / "golden.json", eval_case(id="fueleu-one"))

    with pytest.raises(EmptyError, match="nothing-here"):
        EvalDataset.load(file, case_filter="nothing-here")


def test_load_refuses_a_dataset_with_no_cases(tmp_path: Path) -> None:
    file = _write_dataset(tmp_path / "golden.json")

    with pytest.raises(EmptyError, match="no cases"):
        EvalDataset.load(file)


# Run provenance and summary


def test_the_summary_carries_provenance_and_scores_then_names_the_cases_that_raised() -> None:
    results = (eval_result(), eval_result(eval_case(id="boom"), error="TimeoutError"))
    run = EvalRun(
        dataset_sha="abc",
        case_filter="fueleu",
        settings=get_config_snapshot(EVAL_CONFIG_SECTIONS),
        metrics=compute_metrics(results),
        results=results,
    )

    summary = run.summary()
    body = json.loads(summary.split("\nerrored:")[0])

    assert body["dataset_sha"] == "abc"
    assert body["case_filter"] == "fueleu"
    assert body["settings"]["CHAT_MODEL"] == config.CHAT_MODEL
    assert body["metrics"]["errors"] == 1
    assert "results" not in body
    assert summary.rstrip().endswith("boom  TimeoutError")
