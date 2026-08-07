"""What the embed stage reports: its buckets, its summary line, and how runs combine."""

from app.ingestion.embed.models import EmbedRunResult


def test_a_fresh_result_is_ok_and_empty():
    result = EmbedRunResult()
    assert result.ok
    assert result.counts() == {"embedded": 0, "unchanged": 0}


def test_summary_lists_every_bucket_and_closes_with_failures():
    result = EmbedRunResult(embedded=142, unchanged=1546)
    assert result.summary() == "142 embedded, 1546 unchanged, 0 failed"


def test_a_failure_makes_the_result_not_ok():
    result = EmbedRunResult(embedded=1, failed={"32023R1805": "LLMError: embedding call failed"})
    assert not result.ok
    assert result.summary() == "1 embedded, 0 unchanged, 1 failed"


def test_results_add_field_by_field():
    combined = EmbedRunResult(embedded=2, unchanged=3) + EmbedRunResult(
        embedded=4, failed={"a": "x"}
    )
    assert (combined.embedded, combined.unchanged) == (6, 3)
    assert combined.failed == {"a": "x"}


def test_details_names_the_document_that_failed():
    result = EmbedRunResult(failed={"32023R1805": "LLMError: embedding call failed"})
    assert result.details() == ["failed: 32023R1805 (LLMError: embedding call failed)"]
