"""Ingest reporting: the stage-delta base, the three stage deltas, and the run report."""

import pytest
from pydantic import Field

from app.ingestion.enums import DocAction, IngestRunStatus
from app.ingestion.models import ChunkDelta, FetchDelta, IngestStageDelta, ParseDelta, RunReport


class Delta(IngestStageDelta):
    """A stand-in subclass so the base can be tested without a real stage."""

    added: int = 0
    refs: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"added": self.added, "refs": len(self.refs)}


class OtherDelta(IngestStageDelta):
    """A second subclass with different field names, to prove __add__ rejects a type mismatch."""

    removed: int = 0
    tags: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"removed": self.removed, "tags": len(self.tags)}


class StrFieldDelta(IngestStageDelta):
    """A subclass with an unmergeable str field."""

    note: str = ""

    def counts(self) -> dict[str, int]:
        return {}


class BoolFieldDelta(IngestStageDelta):
    """A subclass with an unmergeable bool field."""

    flag: bool = False

    def counts(self) -> dict[str, int]:
        return {}


def test_ok_is_true_until_something_fails() -> None:
    assert Delta().ok
    assert not Delta(failed={"32023R1805": "ParseError: no body"}).ok


def test_counts_must_be_filled_in_by_the_subclass() -> None:
    with pytest.raises(NotImplementedError):
        IngestStageDelta().summary()


def test_summary_renders_counts_in_order_then_the_failure_count() -> None:
    delta = Delta(added=3, refs=["a", "b"], failed={"c": "boom"})
    assert delta.summary() == "3 added, 2 refs, 1 failed"


def test_summary_always_reports_failures_even_when_there_are_none() -> None:
    assert Delta(added=1).summary() == "1 added, 0 refs, 0 failed"


def test_details_lists_failures_sorted_by_ref() -> None:
    delta = Delta(failed={"b": "second", "a": "first"})
    assert delta.details() == ["failed: a (first)", "failed: b (second)"]


def test_add_sums_ints_concatenates_lists_and_merges_dicts() -> None:
    total = Delta(added=1, refs=["a"], failed={"x": "one"}) + Delta(
        added=2, refs=["b"], failed={"y": "two"}
    )
    assert total.added == 3
    assert total.refs == ["a", "b"]
    assert total.failed == {"x": "one", "y": "two"}


def test_add_returns_the_subclass_not_the_base() -> None:
    assert type(Delta() + Delta()) is Delta


def test_add_leaves_both_operands_untouched() -> None:
    left, right = Delta(added=1, refs=["a"]), Delta(added=2, refs=["b"])
    _ = left + right
    assert (left.added, left.refs) == (1, ["a"])
    assert (right.added, right.refs) == (2, ["b"])


def test_add_rejects_mismatched_subclasses() -> None:
    with pytest.raises(TypeError):
        Delta() + OtherDelta()


def test_add_rejects_a_str_field_naming_it_in_the_error() -> None:
    with pytest.raises(TypeError, match="note"):
        StrFieldDelta(note="a") + StrFieldDelta(note="b")


def test_add_rejects_a_bool_field_rather_than_raising_validation_error() -> None:
    with pytest.raises(TypeError):
        BoolFieldDelta(flag=True) + BoolFieldDelta(flag=True)


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
