"""Evals CLI: exit codes and what `run` prints."""

import pytest

from app.core.config import EVAL_CONFIG_SECTIONS, get_config_snapshot
from app.evals import cli
from app.evals.cli import main
from app.evals.dataset.enums import DriftKind
from app.evals.dataset.models import CaseReference, DriftedReference
from app.evals.metrics import compute_metrics
from app.evals.models import EvalRun
from tests.evals.conftest import eval_case, eval_result


def test_a_subcommand_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])


class _RecordedResults(list):
    """The results a stubbed run returns, plus the judge flag each run was asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.judged: list[bool] = []


@pytest.fixture
def fake_run(monkeypatch):
    """Replace the graph run with a stub returning a chosen list of results."""
    results = _RecordedResults()
    judged = results.judged

    async def _fake_corpus_read(dataset):
        return (), "2026-08-01-a3f1c2"

    async def _fake(dataset, corpus_version=None, stale_cases=(), *, judge=True):
        judged.append(judge)
        chosen = tuple(results)
        return EvalRun(
            dataset_sha=dataset.sha256,
            case_filter=dataset.case_filter,
            corpus_version=corpus_version,
            stale_cases=stale_cases,
            settings=get_config_snapshot(EVAL_CONFIG_SECTIONS),
            metrics=compute_metrics(chosen),
            results=chosen,
        )

    monkeypatch.setattr(cli, "check_against_corpus", _fake_corpus_read)
    monkeypatch.setattr(cli, "evaluate_all_cases", _fake)
    return results


def test_run_judges_unless_told_not_to(fake_run):
    fake_run.append(eval_result())

    main(["run"])
    main(["run", "--no-judge"])

    assert fake_run.judged == [True, False]


def test_run_prints_the_summary_and_exits_zero(fake_run, capsys):
    fake_run.append(eval_result())

    assert main(["run"]) == 0

    out = capsys.readouterr().out
    assert '"raw_recall": 1.0' in out
    assert '"CHAT_MODEL"' in out


def test_run_exits_nonzero_when_a_case_raised(fake_run, capsys):
    fake_run.append(eval_result(eval_case(id="boom"), error="TimeoutError"))

    assert main(["run"]) == 1
    assert "boom  TimeoutError" in capsys.readouterr().out


def test_run_exits_nonzero_when_no_case_matches_the_filter(fake_run, capsys):
    assert main(["run", "--case", "nothing-here"]) == 1
    assert "nothing-here" in capsys.readouterr().out


# Which commands replay their paid calls


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    """Record whether the command turned the call cache on, without turning it on. Autouse
    so no test here installs a real cache: `run` enables one by default, which would put a
    cache under the real data directory and leave it set for whatever runs next."""
    calls: list[bool] = []
    monkeypatch.setattr(cli, "enable_call_cache", lambda: calls.append(True))
    return calls


def test_run_replays_its_embed_and_rerank_calls_by_default(fake_run, enabled):
    fake_run.append(eval_result())

    assert main(["run"]) == 0
    assert enabled


def test_no_cache_makes_a_run_pay_for_its_calls_again(fake_run, enabled):
    fake_run.append(eval_result())

    assert main(["run", "--no-cache"]) == 0
    assert not enabled


def test_run_lists_every_case_only_when_asked(fake_run, capsys):
    fake_run.append(eval_result())

    assert main(["run"]) == 0
    assert "raw 1.00" not in capsys.readouterr().out

    assert main(["run", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "raw 1.00" in out
    assert '"raw_recall": 1.0' in out


def test_run_reports_the_corpus_and_the_stale_cases_it_read_before_scoring(
    fake_run, monkeypatch, capsys
):
    """Tuning compares two runs, so a score has to say which corpus it was measured against
    and which of its reference answers are owed a re-review."""

    async def _fake_corpus_read(dataset):
        moved = CaseReference(celex="32023R1805", article="4")
        drifted = DriftedReference(case_id="amended", target=moved, kind=DriftKind.STALE)
        return (drifted,), "2026-08-01-a3f1c2"

    fake_run.append(eval_result())
    monkeypatch.setattr(cli, "check_against_corpus", _fake_corpus_read)

    assert main(["run"]) == 0

    out = capsys.readouterr().out
    assert '"corpus_version": "2026-08-01-a3f1c2"' in out
    assert "1 case cites text that changed since authoring:" in out
    assert "  amended" in out
