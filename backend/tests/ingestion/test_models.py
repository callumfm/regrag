"""Run report: bucketing, status, and the printed summary."""

from app.ingestion.enums import DocAction, IngestRunStatus
from app.ingestion.models import RunReport


def test_report_record_routes_to_buckets():
    report = RunReport(run_id=1)
    report.record(DocAction.NEW, "a")
    report.record(DocAction.CHANGED, "b")
    report.record(DocAction.UNCHANGED, "c")
    assert (report.new, report.changed, report.unchanged) == (["a"], ["b"], ["c"])


def test_report_ok_and_status_track_failures():
    report = RunReport(run_id=1)
    assert report.ok and report.status is IngestRunStatus.COMPLETED
    report.failed["x"] = "ResolutionError: boom"
    assert not report.ok and report.status is IngestRunStatus.FAILED


def test_report_is_not_ok_when_a_document_failed_to_parse():
    report = RunReport(run_id=1)
    report.unparsed["32023R2449"] = "ParseError: unrecognised EUR-Lex dialect"
    assert not report.ok
    assert report.status is IngestRunStatus.FAILED


def test_summary_counts_and_lists_non_empty_buckets():
    report = RunReport(run_id=7)
    report.record(DocAction.NEW, "32026R0394")
    report.record(DocAction.UNCHANGED, "32015R0757")
    report.dropped.append("32014R0666")
    report.failed["32023R2917"] = "ResolutionError: no fetchable HTML"
    text = report.summary()
    assert "run 7: 1 new, 0 changed, 1 unchanged, 1 dropped, 1 failed" in text
    assert "new: 32026R0394" in text
    assert "dropped: 32014R0666" in text
    assert "failed: 32023R2917 (ResolutionError: no fetchable HTML)" in text
    assert "changed:" not in text


def test_summary_reports_the_chunk_stage():
    report = RunReport(run_id=7, chunks_added=26, chunks_removed=12, chunks_unchanged=774)
    assert "chunks: +26 added, -12 removed, 774 unchanged" in report.summary()


def test_summary_lists_unparsed_documents():
    report = RunReport(run_id=7)
    report.unparsed["32023R2449"] = "ParseError: unrecognised EUR-Lex dialect"
    assert "unparsed: 32023R2449 (ParseError: unrecognised EUR-Lex dialect)" in report.summary()
