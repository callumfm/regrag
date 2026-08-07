"""The shared stage-result base: reporting, and how two results combine."""

from typing import ClassVar

import pytest
from pydantic import Field

from app.ingestion.models import StageRunResult


class Result(StageRunResult):
    """A stand-in subclass so the base can be tested without a real stage."""

    added: int = 0
    celexes: list[str] = Field(default_factory=list)


class OtherResult(StageRunResult):
    """A second subclass with different field names, to prove __add__ rejects a type mismatch."""

    removed: int = 0
    tags: list[str] = Field(default_factory=list)


class StrFieldResult(StageRunResult):
    """A subclass with an unmergeable str field."""

    note: str = ""


class BoolFieldResult(StageRunResult):
    """A subclass with an unmergeable bool field."""

    flag: bool = False


def test_ok_is_true_until_something_fails() -> None:
    assert Result().ok
    assert not Result(failed={"32023R1805": "ParseError: no body"}).ok


def test_counts_covers_every_declared_field_but_the_uncounted_ones() -> None:
    assert Result(added=3, celexes=["a", "b"], failed={"c": "boom"}).counts() == {
        "added": 3,
        "celexes": 2,
    }


def test_counts_skips_a_field_the_subclass_declares_uncounted() -> None:
    class Quiet(Result):
        UNCOUNTED: ClassVar[frozenset[str]] = Result.UNCOUNTED | {"celexes"}

    assert Quiet(added=3, celexes=["a", "b"]).counts() == {"added": 3}


def test_report_pairs_the_counts_with_the_failures() -> None:
    result = Result(added=3, celexes=["a", "b"], failed={"c": "boom"})
    assert result.report() == {"added": 3, "celexes": 2, "failed": {"c": "boom"}}


def test_report_carries_a_new_bucket_without_being_told_about_it() -> None:
    """Adding a field to a stage result must not need a reporting change."""

    class Extra(Result):
        skipped: int = 0

    assert Extra(added=1, skipped=2).report() == {
        "added": 1,
        "celexes": 0,
        "skipped": 2,
        "failed": {},
    }


def test_report_caps_a_failure_message_a_provider_made_too_long() -> None:
    result = Result(failed={"c": "x" * (StageRunResult.MAX_FAILURE_CHARS + 100)})
    assert result.report()["failed"]["c"] == "x" * StageRunResult.MAX_FAILURE_CHARS


def test_report_leaves_the_recorded_failure_message_whole() -> None:
    """Capping is a storage concern: details() still prints the message in full."""
    message = "x" * (StageRunResult.MAX_FAILURE_CHARS + 100)
    result = Result(failed={"c": message})
    result.report()
    assert result.failed["c"] == message


def test_a_stage_with_no_buckets_still_reports_its_failures() -> None:
    assert StageRunResult(failed={"c": "boom"}).summary() == "1 failed"


def test_summary_renders_counts_in_order_then_the_failure_count() -> None:
    result = Result(added=3, celexes=["a", "b"], failed={"c": "boom"})
    assert result.summary() == "3 added, 2 celexes, 1 failed"


def test_summary_always_reports_failures_even_when_there_are_none() -> None:
    assert Result(added=1).summary() == "1 added, 0 celexes, 0 failed"


def test_details_lists_failures_sorted_by_celex() -> None:
    result = Result(failed={"b": "second", "a": "first"})
    assert result.details() == ["failed: a (first)", "failed: b (second)"]


def test_add_sums_ints_concatenates_lists_and_merges_dicts() -> None:
    total = Result(added=1, celexes=["a"], failed={"x": "one"}) + Result(
        added=2, celexes=["b"], failed={"y": "two"}
    )
    assert total.added == 3
    assert total.celexes == ["a", "b"]
    assert total.failed == {"x": "one", "y": "two"}


def test_add_returns_the_subclass_not_the_base() -> None:
    assert type(Result() + Result()) is Result


def test_add_leaves_both_operands_untouched() -> None:
    left, right = Result(added=1, celexes=["a"]), Result(added=2, celexes=["b"])
    _ = left + right
    assert (left.added, left.celexes) == (1, ["a"])
    assert (right.added, right.celexes) == (2, ["b"])


def test_add_rejects_mismatched_subclasses() -> None:
    with pytest.raises(TypeError):
        Result() + OtherResult()


def test_add_rejects_a_str_field_naming_it_in_the_error() -> None:
    with pytest.raises(TypeError, match="note"):
        StrFieldResult(note="a") + StrFieldResult(note="b")


def test_add_rejects_a_bool_field_rather_than_raising_validation_error() -> None:
    with pytest.raises(TypeError):
        BoolFieldResult(flag=True) + BoolFieldResult(flag=True)
