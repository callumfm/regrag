"""The check and stamp subcommands: exit codes and what they print."""

import pytest

from app.evals import cli as evals_cli
from app.evals.cli import main
from app.evals.dataset import cli as dataset_cli
from app.evals.dataset.enums import DriftKind, EvalTrait
from app.evals.dataset.exceptions import EmptyDatasetError
from app.evals.dataset.models import CaseReference, CorpusStamp, DriftedReference, EvalDataset
from tests.evals.conftest import eval_case, eval_dataset

STAMPED = CaseReference(celex="32023R1805", article="4", content_hashes=("b" * 12,))
MOVED = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))
STAMPED_AT = "2026-08-15-2cc038d"


def _loading(dataset: EvalDataset):
    """A stand-in for EvalDataset.load returning a fixed dataset."""
    return classmethod(lambda cls, *args, **kwargs: dataset)


# `evals check`


@pytest.fixture
def fake_check(monkeypatch):
    """Replace the DB read with a stub returning chosen drift, against a stamped dataset."""
    state: dict = {"drifted": (), "current": STAMPED_AT}

    dataset = eval_dataset(eval_case()).model_copy(
        update={"corpus": CorpusStamp(corpus_version=STAMPED_AT, stamped_at="2026-08-28")}
    )
    monkeypatch.setattr(EvalDataset, "load", _loading(dataset))

    async def _fake(loaded):
        return state["drifted"], state["current"]

    def _set(*drifted: DriftedReference, moved_to: str | None = None):
        state.update(drifted=drifted, current=moved_to or STAMPED_AT)

    monkeypatch.setattr(dataset_cli, "check_against_corpus", _fake)
    return _set


def _drifted(case_id: str, kind: DriftKind) -> DriftedReference:
    return DriftedReference(case_id=case_id, target=STAMPED, kind=kind)


def test_check_exits_zero_when_nothing_has_drifted(fake_check, capsys):
    assert main(["check"]) == 0
    assert "resolves and is stamped" in capsys.readouterr().out


def test_check_fails_only_on_an_unresolved_reference(fake_check, capsys):
    fake_check(_drifted("gone-case", DriftKind.UNRESOLVED))

    assert main(["check"]) == 1

    out = capsys.readouterr().out
    assert "unresolved (no stored chunk answers to it):" in out
    assert "gone-case  32023R1805 Article 4" in out


def test_check_names_a_stale_case_without_failing_the_command(fake_check, capsys):
    """A stale case needs a human re-review, not a red build."""
    fake_check(_drifted("amended-case", DriftKind.STALE))

    assert main(["check"]) == 0

    out = capsys.readouterr().out
    assert "stale (cited text changed since authoring):" in out
    assert "amended-case" in out


def test_check_names_an_unstamped_case_without_failing_the_command(fake_check, capsys):
    fake_check(_drifted("new-case", DriftKind.UNSTAMPED))

    assert main(["check"]) == 0
    assert "unstamped (nothing recorded to compare against):" in capsys.readouterr().out


def test_check_groups_the_kinds_worst_first(fake_check, capsys):
    fake_check(
        _drifted("never-stamped", DriftKind.UNSTAMPED),
        _drifted("amended", DriftKind.STALE),
        _drifted("gone", DriftKind.UNRESOLVED),
    )

    assert main(["check"]) == 1

    out = capsys.readouterr().out
    assert out.index("gone") < out.index("amended") < out.index("never-stamped")


def test_check_reports_a_corpus_that_moved_under_the_stamp(fake_check, capsys):
    fake_check(moved_to="2026-09-02-4e81a90")

    assert main(["check"]) == 0
    assert "corpus moved since stamping (now 2026-09-02-4e81a90)" in capsys.readouterr().out


def test_check_reports_an_empty_dataset_without_a_traceback(monkeypatch, capsys):
    def _raise(cls, *args, **kwargs):
        raise EmptyDatasetError("The dataset has no cases")

    monkeypatch.setattr(EvalDataset, "load", classmethod(_raise))

    assert main(["check"]) == 1
    assert "no cases" in capsys.readouterr().out


def test_check_never_touches_the_call_cache(fake_check, monkeypatch):
    """check only asks the corpus what it holds, so it makes no provider call to replay."""
    calls: list[bool] = []
    monkeypatch.setattr(evals_cli, "enable_call_cache", lambda: calls.append(True))

    assert main(["check"]) == 0
    assert not calls


# `evals stamp`


def test_stamp_writes_the_dataset_and_names_the_cases_whose_hashes_moved(monkeypatch, capsys):
    written: list[EvalDataset] = []
    before = eval_dataset(eval_case(id="amended", references=(MOVED,)), eval_case(id="same"))
    after = eval_dataset(eval_case(id="amended", references=(STAMPED,)), eval_case(id="same"))

    async def _fake_stamp(dataset):
        return after

    monkeypatch.setattr(EvalDataset, "load", _loading(before))
    monkeypatch.setattr(dataset_cli, "_stamp_against_corpus", _fake_stamp)
    monkeypatch.setattr(dataset_cli, "save_dataset", lambda dataset: written.append(dataset))

    assert main(["stamp"]) == 0

    out = capsys.readouterr().out
    assert "stamped 2 cases" in out
    assert "hashes changed: amended" in out
    assert "same" not in out.split("hashes changed:")[1]
    assert written == [after]


def test_stamp_says_so_when_nothing_moved(monkeypatch, capsys):
    dataset = eval_dataset(eval_case(references=(STAMPED,)))

    async def _fake_stamp(loaded):
        return dataset

    monkeypatch.setattr(EvalDataset, "load", _loading(dataset))
    monkeypatch.setattr(dataset_cli, "_stamp_against_corpus", _fake_stamp)
    monkeypatch.setattr(dataset_cli, "save_dataset", lambda loaded: None)

    assert main(["stamp"]) == 0
    assert "no hashes changed" in capsys.readouterr().out


def test_stamp_selects_the_cases_carrying_a_trait(monkeypatch, capsys):
    loaded_with: dict = {}
    dataset = eval_dataset(eval_case(references=(STAMPED,), traits=(EvalTrait.MULTI_HOP,)))

    def _load(cls, *args, **kwargs):
        loaded_with.update(kwargs)
        return dataset.model_copy(update=kwargs)

    async def _fake_stamp(loaded):
        return loaded

    monkeypatch.setattr(EvalDataset, "load", classmethod(_load))
    monkeypatch.setattr(dataset_cli, "_stamp_against_corpus", _fake_stamp)
    monkeypatch.setattr(dataset_cli, "save_dataset", lambda loaded: None)

    assert main(["stamp", "--trait", "multi_hop"]) == 0

    assert loaded_with["selection"].trait is EvalTrait.MULTI_HOP
    assert "stamped 1 cases" in capsys.readouterr().out
