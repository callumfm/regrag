"""`evals check`: how a drifted reference is classified, printed, and exited on."""

from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals import cli as evals_cli
from app.evals.cli import main
from app.evals.dataset import check as check_module
from app.evals.dataset.check import DriftedReference, find_drift, find_moved_corpus
from app.evals.dataset.enums import DriftKind
from app.evals.dataset.models import CaseReference, CorpusStamp, EmptyError
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.enums import IngestRunStatus
from app.ingestion.schemas import IngestRun
from tests.evals.conftest import eval_case, eval_dataset, out_of_corpus_case

pytestmark = pytest.mark.anyio

ARTICLE_4 = ("b" * 12,)
"""The first 12 characters of the make_chunk_row default content hash."""

MOVED = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))
STAMPED = CaseReference(celex="32023R1805", article="4", content_hashes=ARTICLE_4)
MISSING = CaseReference(celex="32023R1805", article="999")


@pytest.fixture
async def article_4(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> DocumentChunk:
    """One stored chunk covering the division every factory case cites."""
    chunk = make_chunk_row(ingest_run)
    db_session.add(chunk)
    await db_session.flush()
    return chunk


async def test_a_case_stamped_against_the_text_that_is_stored_has_not_drifted(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    assert await find_drift(db_session, eval_dataset(eval_case(references=(STAMPED,)))) == ()


async def test_a_case_whose_cited_text_changed_is_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """The amendment that keeps retrieving: the division still resolves, so nothing else
    catches it, but the answer was written against text that has since been rewritten."""
    [drifted] = await find_drift(
        db_session, eval_dataset(eval_case(id="amended", references=(MOVED,)))
    )

    assert drifted == DriftedReference(case_id="amended", target=MOVED, kind=DriftKind.STALE)


async def test_a_reference_no_chunk_answers_to_is_unresolved(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    [drifted] = await find_drift(
        db_session,
        eval_dataset(
            eval_case(id="ok", references=(STAMPED,)),
            eval_case(id="gone", references=(MISSING,)),
        ),
    )

    assert (drifted.case_id, drifted.kind) == ("gone", DriftKind.UNRESOLVED)


async def test_a_stamped_reference_that_stopped_resolving_reads_as_unresolved_not_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Hard drift is the louder fact and the one that fails the command, so it wins."""
    stamped_but_gone = CaseReference(celex="32023R1805", article="999", content_hashes=ARTICLE_4)

    [drifted] = await find_drift(
        db_session, eval_dataset(eval_case(references=(stamped_but_gone,)))
    )

    assert drifted.kind is DriftKind.UNRESOLVED


async def test_a_reference_that_was_never_stamped_is_unstamped_rather_than_stale(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Nothing was recorded to compare against, so calling it stale would be an invention."""
    [drifted] = await find_drift(db_session, eval_dataset(eval_case()))

    assert drifted.kind is DriftKind.UNSTAMPED


async def test_out_of_corpus_cases_have_nothing_to_drift(db_session: AsyncSession) -> None:
    assert await find_drift(db_session, eval_dataset(out_of_corpus_case())) == ()


# Whether the corpus itself has moved


def _stamped_at(version: str | None):
    return eval_dataset(eval_case()).model_copy(
        update={"corpus": CorpusStamp(corpus_version=version, stamped_at="2026-08-28")}
    )


@pytest.fixture
async def ingested(db_session: AsyncSession) -> IngestRun:
    """A successful run carrying a corpus version, as the standing corpus would."""
    run = IngestRun(status=IngestRunStatus.SUCCESS, corpus_version="2026-09-02-4e81a90")
    db_session.add(run)
    await db_session.flush()
    return run


async def test_a_dataset_stamped_at_the_current_version_has_not_moved(
    db_session: AsyncSession, ingested: IngestRun
) -> None:
    assert await find_moved_corpus(db_session, _stamped_at("2026-09-02-4e81a90")) is None


async def test_a_dataset_stamped_at_an_older_version_names_the_current_one(
    db_session: AsyncSession, ingested: IngestRun
) -> None:
    assert await find_moved_corpus(db_session, _stamped_at("2026-08-15-2cc038d")) == (
        "2026-09-02-4e81a90"
    )


async def test_an_unstamped_dataset_has_no_corpus_move_to_report(
    db_session: AsyncSession, ingested: IngestRun
) -> None:
    assert await find_moved_corpus(db_session, eval_dataset(eval_case())) is None


# What the command prints and exits


@pytest.fixture
def fake_check(monkeypatch):
    """Replace the DB read with a stub returning chosen drift."""
    state: dict = {"drifted": (), "moved_to": None}

    async def _fake():
        return state["drifted"], state["moved_to"]

    def _set(*drifted: DriftedReference, moved_to: str | None = None):
        state.update(drifted=drifted, moved_to=moved_to)

    monkeypatch.setattr(check_module, "_inspect", _fake)
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
    async def _fake():
        raise EmptyError("The dataset has no cases")

    monkeypatch.setattr(check_module, "_inspect", _fake)

    assert main(["check"]) == 1
    assert "no cases" in capsys.readouterr().out


def test_check_never_touches_the_call_cache(fake_check, monkeypatch):
    """check only asks the corpus what it holds, so it makes no provider call to replay."""
    calls: list[bool] = []
    monkeypatch.setattr(evals_cli, "enable_call_cache", lambda: calls.append(True))

    assert main(["check"]) == 0
    assert not calls
