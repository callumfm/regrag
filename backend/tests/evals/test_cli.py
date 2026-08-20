"""Evals CLI: exit codes and what `check` and `run` print."""

import pytest

from app.evals import cli
from app.evals.cli import format_case_lines, main
from app.evals.metrics import compute_metrics
from app.evals.models import EvalRun, RunSettings, UnresolvedReference
from app.retrieval.models import ReferenceTarget
from tests.evals.conftest import eval_case, eval_result, refused_result


@pytest.fixture
def fake_check(monkeypatch):
    """Replace the DB coroutine with a stub returning a chosen set of unresolved references."""
    unresolved = []

    async def _fake():
        return tuple(unresolved)

    monkeypatch.setattr(cli, "_check_dataset_references", _fake)
    return unresolved


def test_check_exits_zero_when_every_reference_resolves(fake_check, capsys):
    assert main(["check"]) == 0
    assert "resolve" in capsys.readouterr().out


def test_check_names_each_stale_reference_and_exits_nonzero(fake_check, capsys):
    fake_check.append(
        UnresolvedReference(
            case_id="stale-case", target=ReferenceTarget(celex="32023R1805", article="999")
        )
    )

    assert main(["check"]) == 1

    out = capsys.readouterr().out
    assert "stale-case" in out
    assert "32023R1805" in out
    assert "999" in out


def test_a_subcommand_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])


@pytest.fixture
def fake_run(monkeypatch):
    """Replace the graph run with a stub returning a chosen list of results."""
    results: list = []

    async def _fake(dataset, pattern):
        chosen = tuple(results)
        return EvalRun(
            dataset_sha=dataset.sha256,
            case_pattern=pattern,
            settings=RunSettings.from_config(),
            metrics=compute_metrics(chosen),
            results=chosen,
        )

    monkeypatch.setattr(cli, "run_dataset", _fake)
    return results


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


def test_run_exits_nonzero_when_no_case_matches_the_pattern(fake_run, capsys):
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


def test_check_never_touches_the_cache(fake_check, enabled):
    """check only asks the corpus what it holds, so it makes no provider call to replay."""
    assert main(["check"]) == 0
    assert not enabled


# The per-case lines --verbose adds


def test_a_case_line_shows_what_it_retrieved_what_it_cited_and_how_it_ended():
    [line] = format_case_lines((eval_result(),))

    assert line.startswith("case")
    assert "raw 1.00" in line
    assert "exp 1.00" in line
    assert "cite 1.00" in line
    assert "done" in line
    assert "1000ms" in line


def test_a_case_with_nothing_to_recall_prints_dashes_rather_than_zeroes():
    """An out-of-corpus case authors no reference, so its recall is unmeasured; a 0.00
    would read as a retrieval failure on a case that has nothing to retrieve."""
    [line] = format_case_lines((refused_result(),))

    assert "raw    -" in line
    assert "cite    -" in line
    assert "refused" in line


def test_a_case_the_graph_raised_on_scores_nothing_and_is_named_with_its_error():
    """The aggregate leaves an errored case out, so its line must not show scores either."""
    [line] = format_case_lines((eval_result(eval_case(id="boom"), error="TimeoutError"),))

    assert line.startswith("boom")
    assert "raw    -" in line
    assert "error" in line
    assert line.rstrip().endswith("TimeoutError")


def test_the_case_column_is_sized_to_the_longest_id_in_the_run():
    """The ids run from 13 to 44 characters, so a fixed column either wraps or wastes."""
    lines = format_case_lines(
        (eval_result(eval_case(id="short")), eval_result(eval_case(id="a" * 40)))
    )

    assert [line.index("raw") for line in lines] == [42, 42]


def test_run_lists_every_case_only_when_asked(fake_run, capsys):
    fake_run.append(eval_result())

    assert main(["run"]) == 0
    assert "raw 1.00" not in capsys.readouterr().out

    assert main(["run", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "raw 1.00" in out
    assert '"raw_recall": 1.0' in out
