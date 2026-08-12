"""One ingest run's outcome: what it counts, the row it stores, the summary the CLI prints."""

import pytest

from app.ingestion.chunk.models import ChunkCounts
from app.ingestion.constants import MAX_FAILURE_CHARS
from app.ingestion.enums import DocChange, IngestRunStatus, Stage
from app.ingestion.exceptions import ParseError
from app.ingestion.models import DocumentOutcome, IngestRunResult


def failed_doc(
    stage: Stage = Stage.PARSE, celex: str = "b", error: str = "ParseError: no body"
) -> DocumentOutcome:
    """A document outcome as the loop records a stage failure."""
    return DocumentOutcome(celex=celex, failed=stage, error=error)


@pytest.fixture
def run() -> IngestRunResult:
    """A run that fetched two documents and chunked one of them."""
    result = IngestRunResult(run_id=7, corpus_version="2026-08-05-abc1234")
    result.documents.append(
        DocumentOutcome(celex="a", change=DocChange.NEW, chunks=ChunkCounts(added=12, kept=30))
    )
    result.documents.append(
        DocumentOutcome(celex="b", change=DocChange.REUSED, chunks=ChunkCounts())
    )
    return result


def test_a_run_is_ok_when_no_document_failed(run: IngestRunResult) -> None:
    assert run.ok
    assert run.status is IngestRunStatus.SUCCESS


@pytest.mark.parametrize("stage", [Stage.FETCH, Stage.PARSE, Stage.CHUNK])
def test_a_failure_in_any_stage_fails_the_run(stage: Stage) -> None:
    result = IngestRunResult(run_id=1)
    result.documents.append(failed_doc(stage))
    assert not result.ok
    assert result.status is IngestRunStatus.FAILED


def test_an_embed_failure_fails_the_run() -> None:
    result = IngestRunResult(run_id=1)
    result.embed.fail("a", ParseError("boom"), chunks=1)
    assert not result.ok


def test_a_failed_document_counts_towards_nothing_but_its_failure() -> None:
    """The loop rolled its work back, so no stage may claim it."""
    result = IngestRunResult(run_id=1)
    result.documents.append(failed_doc(Stage.CHUNK, "a"))

    assert result.report()["fetch"] == {"new": 0, "updated": 0, "reused": 0, "failed": {}}
    assert result.report()["parse"] == {"parsed": 0, "failed": {}}
    assert result.report()["chunk"]["failed"] == {"a": "ParseError: no body"}


def test_the_prune_is_counted_as_chunks_deleted(run: IngestRunResult) -> None:
    run.pruned = 5
    assert run.chunks == ChunkCounts(added=12, deleted=5, kept=30)


def test_a_run_with_no_fetch_or_parse_failure_may_prune(run: IngestRunResult) -> None:
    assert run.corpus_complete
    run.documents.append(failed_doc(Stage.CHUNK, "c"))
    assert run.corpus_complete


@pytest.mark.parametrize("stage", [Stage.FETCH, Stage.PARSE])
def test_a_run_that_lost_a_document_may_not_prune(run: IngestRunResult, stage: Stage) -> None:
    run.documents.append(failed_doc(stage, "c"))
    assert not run.corpus_complete


def test_report_covers_every_stage_with_its_counts_and_failures(run: IngestRunResult) -> None:
    run.dropped = ["z"]
    run.documents.append(failed_doc(Stage.PARSE, "c"))
    run.embed.embedded = 12

    assert run.report() == {
        "discover": {"dropped": 1, "failed": {}},
        "fetch": {"new": 1, "updated": 0, "reused": 1, "failed": {}},
        "parse": {"parsed": 2, "failed": {"c": "ParseError: no body"}},
        "chunk": {"added": 12, "deleted": 0, "kept": 30, "refreshed": 0, "failed": {}},
        "embed": {"embedded": 12, "already_embedded": 0, "failed": {}},
    }


def test_report_leaves_out_the_run_s_own_columns(run: IngestRunResult) -> None:
    """Both are columns already, and corpus_version is stamped after the row is written."""
    assert "run_id" not in run.report()
    assert "corpus_version" not in run.report()


def test_report_caps_a_failure_message_a_provider_made_too_long() -> None:
    result = IngestRunResult(run_id=1)
    result.embed.fail("c", ParseError("x" * (MAX_FAILURE_CHARS + 100)), chunks=1)
    stored = result.report()["embed"]["failed"]["c"]
    assert len(stored) == MAX_FAILURE_CHARS
    assert stored.startswith("1 chunk: ParseError: xxx")


def test_report_leaves_the_recorded_failure_message_whole() -> None:
    """Capping is a storage concern: the summary still prints the message in full."""
    message = "x" * (MAX_FAILURE_CHARS + 100)
    result = IngestRunResult(run_id=1)
    result.embed.fail("c", ParseError(message), chunks=1)
    result.report()
    assert message in result.summary()


def test_the_discover_line_reports_the_given_total_not_the_documents_seen_so_far() -> None:
    """The mid-run discover log fires before documents populate, so it takes an explicit total."""
    result = IngestRunResult(run_id=1, dropped=["z"])
    assert result.line(Stage.DISCOVER, total=2) == "[discover] 2 documents: 1 dropped, 0 failed"


def test_summary_reports_every_stage_on_its_own_line_with_its_unit(run: IngestRunResult) -> None:
    run.embed.embedded, run.embed.already_embedded = 12, 30

    assert run.summary().splitlines() == [
        "run 7 (2026-08-05-abc1234)",
        "  [discover] 2 documents: 0 dropped, 0 failed",
        "  [fetch] 2 documents: 1 new, 0 updated, 1 reused, 0 failed",
        "  [parse] 2 documents: 2 parsed, 0 failed",
        "  [chunk] 42 chunks: 12 added, 0 deleted, 30 kept, 0 refreshed, 0 failed",
        "  [embed] 42 chunks: 12 embedded, 30 already embedded, 0 failed",
        "  fetch new: a",
    ]


def test_summary_says_so_when_no_version_was_stamped() -> None:
    assert IngestRunResult(run_id=7).summary().startswith("run 7 (not stamped)")


def test_summary_lists_what_discovery_dropped_and_what_each_stage_failed() -> None:
    result = IngestRunResult(run_id=7, dropped=["z"])
    result.documents.append(failed_doc(Stage.FETCH, "a", "ConnectionError: 404"))
    result.documents.append(failed_doc(Stage.PARSE, "b"))
    result.embed.fail("c", ParseError("provider down"), chunks=1)

    assert "  discover dropped: z" in result.summary()
    assert "  fetch failed: a (ConnectionError: 404)" in result.summary()
    assert "  parse failed: b (ParseError: no body)" in result.summary()
    assert "  embed failed: c (1 chunk: ParseError: provider down)" in result.summary()


def test_refreshed_chunks_are_summed_across_documents(run: IngestRunResult) -> None:
    run.documents.append(
        DocumentOutcome(celex="d", change=DocChange.UPDATED, chunks=ChunkCounts(refreshed=4))
    )
    assert run.chunks.refreshed == 4
