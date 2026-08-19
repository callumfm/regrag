import json

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import ChatConfig, EmbeddingConfig, RetrievalConfig, config
from app.evals.enums import EvalKind
from app.evals.metrics import compute_metrics
from app.evals.models import EvalCase, EvalDataset, EvalRun, RunSettings
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


# Run provenance and summary


def test_run_settings_record_every_setting_the_run_reads(monkeypatch) -> None:
    """Taken from the config sections whole, so a knob added later is recorded without
    anyone editing the model."""
    monkeypatch.setattr(config, "CHAT_CONTEXT_CHUNKS", 7)

    settings = RunSettings.from_config().root

    sections = (EmbeddingConfig, ChatConfig, RetrievalConfig)
    expected = {name for section in sections for name in section.model_fields}
    assert set(settings) == expected - {"VOYAGE_API_KEY", "ANTHROPIC_API_KEY"}
    assert settings["CHAT_CONTEXT_CHUNKS"] == 7
    assert settings["RERANK_POOL"] == config.RERANK_POOL


def test_run_settings_leave_out_the_secrets(monkeypatch) -> None:
    """Provenance is printed and pasted around; a key is not a knob a run reproduces."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", SecretStr("sk-never-recorded"))

    dumped = RunSettings.from_config().model_dump_json()

    assert "sk-never-recorded" not in dumped
    assert "API_KEY" not in dumped


def test_the_summary_carries_provenance_and_scores_then_names_the_cases_that_raised() -> None:
    results = (eval_result(), eval_result(eval_case(id="boom"), error="TimeoutError"))
    run = EvalRun(
        dataset_sha="abc",
        case_pattern="fueleu",
        settings=RunSettings.from_config(),
        metrics=compute_metrics(results),
        results=results,
    )

    summary = run.summary()
    body = json.loads(summary.split("\nerrored:")[0])

    assert body["dataset_sha"] == "abc"
    assert body["case_pattern"] == "fueleu"
    assert body["settings"]["CHAT_MODEL"] == config.CHAT_MODEL
    assert body["metrics"]["errors"] == 1
    assert "results" not in body
    assert summary.rstrip().endswith("boom  TimeoutError")
