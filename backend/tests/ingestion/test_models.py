"""The shared stage-result base: reporting, and how two results combine."""

import pytest
from pydantic import Field

from app.ingestion.models import StageRunResult


class Result(StageRunResult):
    """A stand-in subclass so the base can be tested without a real stage."""

    added: int = 0
    refs: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"added": self.added, "refs": len(self.refs)}


class OtherResult(StageRunResult):
    """A second subclass with different field names, to prove __add__ rejects a type mismatch."""

    removed: int = 0
    tags: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"removed": self.removed, "tags": len(self.tags)}


class StrFieldResult(StageRunResult):
    """A subclass with an unmergeable str field."""

    note: str = ""

    def counts(self) -> dict[str, int]:
        return {}


class BoolFieldResult(StageRunResult):
    """A subclass with an unmergeable bool field."""

    flag: bool = False

    def counts(self) -> dict[str, int]:
        return {}


def test_ok_is_true_until_something_fails() -> None:
    assert Result().ok
    assert not Result(failed={"32023R1805": "ParseError: no body"}).ok


def test_counts_must_be_filled_in_by_the_subclass() -> None:
    with pytest.raises(NotImplementedError):
        StageRunResult().summary()


def test_summary_renders_counts_in_order_then_the_failure_count() -> None:
    result = Result(added=3, refs=["a", "b"], failed={"c": "boom"})
    assert result.summary() == "3 added, 2 refs, 1 failed"


def test_summary_always_reports_failures_even_when_there_are_none() -> None:
    assert Result(added=1).summary() == "1 added, 0 refs, 0 failed"


def test_details_lists_failures_sorted_by_ref() -> None:
    result = Result(failed={"b": "second", "a": "first"})
    assert result.details() == ["failed: a (first)", "failed: b (second)"]


def test_add_sums_ints_concatenates_lists_and_merges_dicts() -> None:
    total = Result(added=1, refs=["a"], failed={"x": "one"}) + Result(
        added=2, refs=["b"], failed={"y": "two"}
    )
    assert total.added == 3
    assert total.refs == ["a", "b"]
    assert total.failed == {"x": "one", "y": "two"}


def test_add_returns_the_subclass_not_the_base() -> None:
    assert type(Result() + Result()) is Result


def test_add_leaves_both_operands_untouched() -> None:
    left, right = Result(added=1, refs=["a"]), Result(added=2, refs=["b"])
    _ = left + right
    assert (left.added, left.refs) == (1, ["a"])
    assert (right.added, right.refs) == (2, ["b"])


def test_add_rejects_mismatched_subclasses() -> None:
    with pytest.raises(TypeError):
        Result() + OtherResult()


def test_add_rejects_a_str_field_naming_it_in_the_error() -> None:
    with pytest.raises(TypeError, match="note"):
        StrFieldResult(note="a") + StrFieldResult(note="b")


def test_add_rejects_a_bool_field_rather_than_raising_validation_error() -> None:
    with pytest.raises(TypeError):
        BoolFieldResult(flag=True) + BoolFieldResult(flag=True)
