"""The `check` and `stamp` subcommands: what they print, and what they exit."""

import pytest

from app.evals import cli as evals_cli
from app.evals.cli import main
from app.evals.dataset import cli
from app.evals.dataset.models import (
    CaseReference,
    ChangedDocument,
    DatasetDrift,
    EmptyError,
    EvalDataset,
    StaleReference,
    UnresolvedReference,
)
from tests.evals.conftest import eval_case, eval_dataset

TARGET = CaseReference(celex="32023R1805", article="4")
MISSING = CaseReference(celex="32023R1805", article="999")


@pytest.fixture
def fake_drift(monkeypatch):
    """Replace the DB read with a stub returning a chosen set of drift signals."""
    drift = DatasetDrift()

    async def _fake():
        return drift

    def _set(**fields):
        nonlocal drift
        drift = DatasetDrift(**fields)

    monkeypatch.setattr(cli, "_inspect", _fake)
    return _set


def test_check_exits_zero_when_nothing_has_drifted(fake_drift, capsys):
    assert main(["check"]) == 0
    assert "resolves and is stamped" in capsys.readouterr().out


def test_check_names_each_unresolved_reference_and_exits_nonzero(fake_drift, capsys):
    fake_drift(unresolved=(UnresolvedReference(case_id="gone-case", target=MISSING),))

    assert main(["check"]) == 1

    out = capsys.readouterr().out
    assert "gone-case" in out
    assert "32023R1805" in out
    assert "999" in out


def test_check_names_a_stale_case_without_failing_the_command(fake_drift, capsys):
    """A stale case needs a human re-review, not a red build, so it is reported and passed."""
    fake_drift(
        stale=(
            StaleReference(
                case_id="amended-case", target=TARGET, stamped=("0" * 12,), current=("a" * 12,)
            ),
        )
    )

    assert main(["check"]) == 0

    out = capsys.readouterr().out
    assert "stale (cited text changed since authoring):" in out
    assert "amended-case" in out
    assert "Article 4" in out


def test_check_names_an_amended_act_with_both_of_its_hashes(fake_drift, capsys):
    fake_drift(
        changed_documents=(ChangedDocument(celex="32023R1805", stamped="9f2c", current="4e81"),)
    )

    assert main(["check"]) == 0
    assert "32023R1805 changed since stamping (9f2c -> 4e81)" in capsys.readouterr().out


def test_check_says_when_an_act_has_left_the_corpus(fake_drift, capsys):
    fake_drift(
        changed_documents=(ChangedDocument(celex="32023R1805", stamped="9f2c", current=None),)
    )

    assert main(["check"]) == 0
    assert "-> gone" in capsys.readouterr().out


def test_check_counts_the_cases_no_stamp_was_ever_recorded_for(fake_drift, capsys):
    fake_drift(unstamped=("new-case", "another"))

    assert main(["check"]) == 0
    assert "2 cases unstamped: new-case, another" in capsys.readouterr().out


def test_check_reports_an_empty_dataset_without_a_traceback(monkeypatch, capsys):
    async def _fake():
        raise EmptyError("The dataset has no cases")

    monkeypatch.setattr(cli, "_inspect", _fake)

    assert main(["check"]) == 1
    assert "no cases" in capsys.readouterr().out


def test_check_never_touches_the_call_cache(fake_drift, monkeypatch):
    """check only asks the corpus what it holds, so it makes no provider call to replay."""
    calls: list[bool] = []
    monkeypatch.setattr(evals_cli, "enable_call_cache", lambda: calls.append(True))

    assert main(["check"]) == 0
    assert not calls


# Stamping


def test_stamp_writes_the_dataset_and_names_the_cases_whose_hashes_moved(monkeypatch, capsys):
    written: list[EvalDataset] = []

    async def _fake(case_filter):
        before = eval_dataset(eval_case(id="amended"), eval_case(id="untouched"))
        after = eval_dataset(
            eval_case(
                id="amended",
                references=(TARGET.model_copy(update={"content_hashes": ("a" * 12,)}),),
            ),
            eval_case(id="untouched"),
        )
        return before, after

    monkeypatch.setattr(cli, "_stamp", _fake)
    monkeypatch.setattr(EvalDataset, "save", lambda self: written.append(self))

    assert main(["stamp"]) == 0

    out = capsys.readouterr().out
    assert "stamped 2 cases" in out
    assert "hashes changed: amended" in out
    assert "untouched" not in out.split("hashes changed:")[1]
    assert len(written) == 1


def test_stamp_says_so_when_nothing_moved(monkeypatch, capsys):
    async def _fake(case_filter):
        dataset = eval_dataset(eval_case())
        return dataset, dataset

    monkeypatch.setattr(cli, "_stamp", _fake)
    monkeypatch.setattr(EvalDataset, "save", lambda self: None)

    assert main(["stamp"]) == 0
    assert "no hashes changed" in capsys.readouterr().out
