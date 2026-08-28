"""`evals stamp`: what it records, what it leaves alone, and how it writes the file."""

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.evals.cli import main
from app.evals.dataset import stamp as stamp_module
from app.evals.dataset.check import find_drift
from app.evals.dataset.enums import EvalKind
from app.evals.dataset.models import CaseReference, CorpusStamp, EvalCase, EvalDataset
from app.evals.dataset.stamp import format_dataset, save_dataset, stamp_dataset
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.enums import IngestRunStatus
from app.ingestion.schemas import IngestRun
from tests.evals.conftest import eval_case, eval_dataset

pytestmark = pytest.mark.anyio

ARTICLE_4 = ("b" * 12,)
"""The first 12 characters of the make_chunk_row default content hash."""

MOVED = CaseReference(celex="32023R1805", article="4", content_hashes=("0" * 12,))
STAMPED = CaseReference(celex="32023R1805", article="4", content_hashes=ARTICLE_4)


@pytest.fixture
async def article_4(
    db_session: AsyncSession, make_chunk_row: Callable[..., DocumentChunk]
) -> DocumentChunk:
    """One stored chunk covering the division every factory case cites, under a run the
    corpus version can be read off."""
    run = IngestRun(status=IngestRunStatus.SUCCESS, corpus_version="2026-09-02-4e81a90")
    db_session.add(run)
    await db_session.flush()
    chunk = make_chunk_row(run)
    db_session.add(chunk)
    await db_session.flush()
    return chunk


async def test_stamping_records_what_each_cited_division_hashes_to_now(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    stamped = await stamp_dataset(db_session, eval_dataset(eval_case()))

    assert stamped.cases[0].references[0].content_hashes == ARTICLE_4
    assert stamped.corpus is not None
    assert stamped.corpus.corpus_version == "2026-09-02-4e81a90"


async def test_stamping_clears_the_drift_it_recorded(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    dataset = eval_dataset(eval_case(references=(MOVED,)))

    stamped = await stamp_dataset(db_session, dataset)

    assert await find_drift(db_session, stamped) == ()


async def test_a_filtered_stamp_leaves_the_cases_it_did_not_select_alone(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """Repairing one case must not silently clear the staleness of every other one."""
    dataset = eval_dataset(
        eval_case(id="fueleu-repaired", references=(MOVED,)),
        eval_case(id="mrv-untouched", references=(MOVED,)),
        case_filter="fueleu",
    )

    stamped = await stamp_dataset(db_session, dataset)

    assert stamped.cases[0].references[0].content_hashes == ARTICLE_4
    assert stamped.cases[1].references[0].content_hashes == ("0" * 12,)


async def test_a_filtered_stamp_leaves_the_corpus_stamp_alone(
    db_session: AsyncSession, article_4: DocumentChunk
) -> None:
    """The stamp covers the whole dataset, so only an unfiltered stamp can honestly claim it."""
    was = CorpusStamp(corpus_version="2026-08-15-2cc038d", stamped_at="2026-08-28")
    dataset = eval_dataset(eval_case(), case_filter="case").model_copy(update={"corpus": was})

    stamped = await stamp_dataset(db_session, dataset)

    assert stamped.corpus == was


# Writing the file back out


def test_a_saved_dataset_loads_back_as_the_same_cases_and_stamp(tmp_path: Path) -> None:
    corpus = CorpusStamp(corpus_version="2026-08-15-2cc038d", stamped_at="2026-08-28")
    dataset = eval_dataset(
        eval_case(id="stamped", references=(STAMPED,)),
        EvalCase(id="ooc", kind=EvalKind.OUT_OF_CORPUS, question="q?"),
    ).model_copy(update={"corpus": corpus})
    file = tmp_path / "golden.json"

    save_dataset(dataset, file)
    loaded = EvalDataset.load(file)

    assert loaded.corpus == corpus
    assert loaded.cases[0].references == (STAMPED,)
    assert loaded.cases[1].references == ()


def test_saving_keeps_a_reference_on_one_line_and_omits_what_it_does_not_carry(
    tmp_path: Path,
) -> None:
    """The file is hand-authored and PR-reviewed, so a re-stamp has to diff as the one line
    that moved rather than reformatting every case around it."""
    file = tmp_path / "golden.json"

    save_dataset(eval_dataset(eval_case(references=(STAMPED,))), file)

    text = file.read_text()
    assert f'{{"celex": "32023R1805", "article": "4", "content_hashes": ["{"b" * 12}"]}}' in text
    assert '"annex"' not in text
    assert '"case_filter"' not in text


def test_the_committed_dataset_is_what_the_writer_produces() -> None:
    """Otherwise the next stamp reformats the whole file and buries what actually moved."""
    assert format_dataset(EvalDataset.load()) == config.EVAL_DATASET_PATH.read_text()


# What the command prints


def test_stamp_writes_the_dataset_and_names_the_cases_whose_hashes_moved(monkeypatch, capsys):
    written: list[EvalDataset] = []

    async def _fake(case_filter):
        before = eval_dataset(eval_case(id="amended", references=(MOVED,)), eval_case(id="same"))
        after = eval_dataset(eval_case(id="amended", references=(STAMPED,)), eval_case(id="same"))
        return before, after

    monkeypatch.setattr(stamp_module, "_stamp", _fake)
    monkeypatch.setattr(stamp_module, "save_dataset", lambda dataset: written.append(dataset))

    assert main(["stamp"]) == 0

    out = capsys.readouterr().out
    assert "stamped 2 cases" in out
    assert "hashes changed: amended" in out
    assert "same" not in out.split("hashes changed:")[1]
    assert len(written) == 1


def test_stamp_says_so_when_nothing_moved(monkeypatch, capsys):
    async def _fake(case_filter):
        dataset = eval_dataset(eval_case(references=(STAMPED,)))
        return dataset, dataset

    monkeypatch.setattr(stamp_module, "_stamp", _fake)
    monkeypatch.setattr(stamp_module, "save_dataset", lambda dataset: None)

    assert main(["stamp"]) == 0
    assert "no hashes changed" in capsys.readouterr().out
