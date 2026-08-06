"""The run report: one result per stage, and how the CLI prints them."""

from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.models import IngestRunResult
from app.ingestion.parse.models import ParseRunResult


def test_stages_finds_every_result_and_nothing_else() -> None:
    assert list(IngestRunResult(run_id=1).stages) == ["fetch", "parse", "chunk"]


def test_a_run_is_ok_when_no_stage_failed() -> None:
    assert IngestRunResult(run_id=1).ok
    assert IngestRunResult(run_id=1).status is IngestRunStatus.COMPLETED


def test_a_failure_in_any_stage_fails_the_run() -> None:
    assert not IngestRunResult(run_id=1, fetch=FetchRunResult(failed={"a": "404"})).ok
    assert not IngestRunResult(run_id=1, parse=ParseRunResult(failed={"a": "ParseError"})).ok
    assert IngestRunResult(run_id=1, chunk=ChunkRunResult(failed={"a": "boom"})).status is (
        IngestRunStatus.FAILED
    )


def test_summary_reports_every_stage_on_its_own_line() -> None:
    result = IngestRunResult(
        run_id=7,
        corpus_version="2026-08-05-abc1234",
        fetch=FetchRunResult(new=["a"], unchanged=["b"]),
        parse=ParseRunResult(parsed=["a", "b"]),
        chunk=ChunkRunResult(added=12, unchanged=30),
    )
    assert result.summary().splitlines() == [
        "run 7 (2026-08-05-abc1234)",
        "  [fetch] 1 new, 0 changed, 1 unchanged, 0 dropped, 0 failed",
        "  [parse] 2 parsed, 0 failed",
        "  [chunk] 12 added, 0 removed, 30 unchanged, 0 failed",
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
