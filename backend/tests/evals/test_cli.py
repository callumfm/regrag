"""Evals CLI: exit codes and what `check` prints."""

import pytest

from app.evals import cli
from app.evals.cli import main
from app.retrieval.models import ReferenceTarget


@pytest.fixture
def fake_check(monkeypatch):
    """Replace the DB coroutine with a stub returning a chosen set of unresolved references."""
    unresolved = []

    async def _fake():
        return tuple(unresolved)

    monkeypatch.setattr(cli, "_check", _fake)
    return unresolved


def test_check_exits_zero_when_every_reference_resolves(fake_check, capsys):
    assert main(["check"]) == 0
    assert "resolve" in capsys.readouterr().out


def test_check_names_each_stale_reference_and_exits_nonzero(fake_check, capsys):
    fake_check.append(("stale-case", ReferenceTarget(celex="32023R1805", article="999")))

    assert main(["check"]) == 1

    out = capsys.readouterr().out
    assert "stale-case" in out
    assert "32023R1805" in out
    assert "999" in out


def test_a_subcommand_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])
