"""Run report: bucketing, status, and the printed summary."""

from app.ingestion.chunk.models import ChunkDelta
from app.ingestion.enums import DocAction, IngestRunStatus
from app.ingestion.fetch.models import FetchDelta
from app.ingestion.models import RunReport
from app.ingestion.parse.models import ParseDelta


def test_record_routes_each_action_to_its_bucket() -> None:
    delta = FetchDelta()
    delta.record(DocAction.NEW, "a")
    delta.record(DocAction.CHANGED, "b")
    delta.record(DocAction.UNCHANGED, "c")
    assert (delta.new, delta.changed, delta.unchanged) == (["a"], ["b"], ["c"])


def test_stages_finds_every_delta_and_nothing_else() -> None:
    assert list(RunReport(run_id=1).stages) == ["fetch", "parse", "chunk"]


def test_a_run_is_ok_when_no_stage_failed() -> None:
    assert RunReport(run_id=1).ok
    assert RunReport(run_id=1).status is IngestRunStatus.COMPLETED


def test_a_failure_in_any_stage_fails_the_run() -> None:
    assert not RunReport(run_id=1, fetch=FetchDelta(failed={"a": "404"})).ok
    assert not RunReport(run_id=1, parse=ParseDelta(failed={"a": "ParseError"})).ok
    assert RunReport(run_id=1, chunk=ChunkDelta(failed={"a": "boom"})).status is (
        IngestRunStatus.FAILED
    )


def test_summary_reports_every_stage_on_its_own_line() -> None:
    report = RunReport(
        run_id=7,
        corpus_version="2026-08-05-abc1234",
        fetch=FetchDelta(new=["a"], unchanged=["b"]),
        parse=ParseDelta(parsed=["a", "b"]),
        chunk=ChunkDelta(added=12, unchanged=30),
    )
    assert report.summary().splitlines() == [
        "run 7 (2026-08-05-abc1234)",
        "  [fetch] 1 new, 0 changed, 1 unchanged, 0 dropped, 0 failed",
        "  [parse] 2 parsed, 0 failed",
        "  [chunk] 12 added, 0 removed, 30 unchanged, 0 failed",
        "  fetch new: a",
    ]


def test_summary_says_so_when_no_version_was_stamped() -> None:
    assert RunReport(run_id=7).summary().startswith("run 7 (not stamped)")


def test_summary_lists_each_stage_s_failures() -> None:
    report = RunReport(
        run_id=7,
        fetch=FetchDelta(failed={"a": "HTTPError: 404"}),
        parse=ParseDelta(failed={"b": "ParseError: no body"}),
    )
    assert "  fetch failed: a (HTTPError: 404)" in report.summary()
    assert "  parse failed: b (ParseError: no body)" in report.summary()
