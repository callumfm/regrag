"""Fetch decision logic: version-ref classification, drop detection, run report."""

from app.ingestion.discover import DocumentSpec
from app.ingestion.fetch import DocAction, RunReport, classify, dropped_refs


def spec(ref, topic="mrv"):
    return DocumentSpec(topic=topic, source="eurlex", ref=ref, candidate_ref=None)


def test_classify_no_baseline_is_new():
    assert classify(None, "32023R2449") is DocAction.NEW


def test_classify_differing_resolved_ref_is_changed():
    assert classify("02015R0757-20240101", "02015R0757-20250101") is DocAction.CHANGED


def test_classify_same_resolved_ref_is_unchanged():
    assert classify("02015R0757-20250101", "02015R0757-20250101") is DocAction.UNCHANGED


def test_dropped_refs_are_baseline_refs_absent_from_discovery():
    specs = [spec("32015R0757"), spec("32016R1928")]
    baseline = ["32015R0757", "32016R1928", "32014R0666"]
    assert dropped_refs(specs, baseline) == ["32014R0666"]


def test_dropped_refs_empty_when_all_discovered():
    assert dropped_refs([spec("32015R0757")], ["32015R0757"]) == []


def test_report_record_routes_to_buckets():
    report = RunReport(run_id=1)
    report.record(DocAction.NEW, "a")
    report.record(DocAction.CHANGED, "b")
    report.record(DocAction.UNCHANGED, "c")
    assert (report.new, report.changed, report.unchanged) == (["a"], ["b"], ["c"])


def test_report_ok_iff_no_failures():
    report = RunReport(run_id=1)
    assert report.ok
    report.failed["x"] = "ResolutionError: boom"
    assert not report.ok


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
