"""Fetch-stage values: how a fetch run records what it did to each document."""

from app.ingestion.enums import DocChange
from app.ingestion.fetch.models import FetchRunResult


def test_record_routes_each_action_to_its_bucket() -> None:
    result = FetchRunResult()
    result.record(DocChange.NEW, "a")
    result.record(DocChange.CHANGED, "b")
    result.record(DocChange.UNCHANGED, "c")
    assert (result.new, result.changed, result.unchanged) == (["a"], ["b"], ["c"])
