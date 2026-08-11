"""One ingest run's outcome: stage roll-up, the row it stores, the summary the CLI prints."""

from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.discover.models import DiscoverRunResult
from app.ingestion.embed.models import EmbedRunResult
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.parse.models import ParseRunResult
from app.ingestion.result import IngestRunResult
from tests.conftest import recorded_run


def test_stages_finds_every_result_and_nothing_else() -> None:
    assert list(IngestRunResult(run_id=1).stages) == [
        "discover",
        "fetch",
        "parse",
        "chunk",
        "embed",
    ]


def test_a_run_is_ok_when_no_stage_failed() -> None:
    assert recorded_run().ok
    assert recorded_run().status is IngestRunStatus.SUCCESS


def test_a_failure_in_any_stage_fails_the_run() -> None:
    assert not recorded_run(fetch=FetchRunResult(failed={"a": "404"})).ok
    assert not recorded_run(parse=ParseRunResult(failed={"a": "ParseError"})).ok
    assert recorded_run(chunk=ChunkRunResult(failed={"a": "boom"})).status is (
        IngestRunStatus.FAILED
    )


def test_a_stage_that_never_reported_fails_the_run() -> None:
    """An empty stage result is ok, so a run cut short must not close as COMPLETED."""
    cut_short = IngestRunResult(run_id=1, started={"fetch"}, fetch=FetchRunResult(new=["a"]))
    assert cut_short.unrecorded == frozenset({"discover", "parse", "chunk", "embed"})
    assert not cut_short.ok
    assert cut_short.status is IngestRunStatus.FAILED


def test_a_stage_that_never_reported_is_null_in_the_report_and_not_run_in_the_summary() -> None:
    cut_short = IngestRunResult(run_id=1, started={"fetch"}, fetch=FetchRunResult(new=["a"]))
    assert cut_short.report()["parse"] is None
    assert cut_short.report()["fetch"]["new"] == 1
    assert "[parse] not run" in cut_short.summary()


def test_summary_reports_every_stage_on_its_own_line() -> None:
    result = recorded_run(
        7,
        corpus_version="2026-08-05-abc1234",
        discover=DiscoverRunResult(),
        fetch=FetchRunResult(new=["a"], unchanged=["b"]),
        parse=ParseRunResult(parsed=["a", "b"]),
        chunk=ChunkRunResult(added=12, unchanged=30),
        embed=EmbedRunResult(embedded=12, unchanged=30),
    )
    assert result.summary().splitlines() == [
        "run 7 (2026-08-05-abc1234)",
        "  [discover] 0 dropped, 0 failed",
        "  [fetch] 1 new, 0 changed, 1 unchanged, 0 failed",
        "  [parse] 2 parsed, 0 failed",
        "  [chunk] 12 added, 0 removed, 30 unchanged, 0 failed",
        "  [embed] 12 embedded, 30 unchanged, 0 failed",
        "  fetch new: a",
    ]


def test_summary_says_so_when_no_version_was_stamped() -> None:
    assert IngestRunResult(run_id=7).summary().startswith("run 7 (not stamped)")


def test_summary_lists_each_stage_s_failures() -> None:
    result = IngestRunResult(
        run_id=7,
        fetch=FetchRunResult(failed={"a": "HTTPError: 404"}),
        parse=ParseRunResult(failed={"b": "ParseError: no body"}),
    )
    assert "  fetch failed: a (HTTPError: 404)" in result.summary()
    assert "  parse failed: b (ParseError: no body)" in result.summary()


def test_report_covers_every_stage_with_its_counts_and_failures() -> None:
    result = recorded_run(
        7,
        discover=DiscoverRunResult(),
        fetch=FetchRunResult(new=["a"], unchanged=["b"]),
        parse=ParseRunResult(parsed=["a"], failed={"b": "ParseError: no body"}),
        chunk=ChunkRunResult(added=12, unchanged=30),
        embed=EmbedRunResult(embedded=12),
    )
    assert result.report() == {
        "discover": {"dropped": 0, "failed": {}},
        "fetch": {"new": 1, "changed": 0, "unchanged": 1, "failed": {}},
        "parse": {"parsed": 1, "failed": {"b": "ParseError: no body"}},
        "chunk": {"added": 12, "removed": 0, "unchanged": 30, "failed": {}},
        "embed": {"embedded": 12, "unchanged": 0, "failed": {}},
    }


def test_report_leaves_out_the_run_s_own_columns() -> None:
    """Both are columns already, and corpus_version is stamped after the row is written."""
    report = IngestRunResult(run_id=7, corpus_version="2026-08-05-abc1234").report()
    assert "run_id" not in report
    assert "corpus_version" not in report
    assert "started" not in report


def test_a_stage_assigned_without_being_marked_still_reads_as_never_run() -> None:
    """Assignment is not evidence a stage ran; only the stage saying so is."""
    built = IngestRunResult(run_id=1, fetch=FetchRunResult(new=["a"]))
    assert "fetch" in built.unrecorded
    assert built.report()["fetch"] is None

    built.mark_reported("fetch")
    assert "fetch" not in built.unrecorded
    assert built.report()["fetch"]["new"] == 1
