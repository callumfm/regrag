"""What a model call spends: how usages add, and what a run with none reads as."""

from langchain_core.messages.ai import UsageMetadata

from app.core.llm.models import TokenUsage

SMALL = TokenUsage(input_tokens=100, output_tokens=10)
TINY = TokenUsage(input_tokens=5, output_tokens=1)


def test_usages_add_field_by_field() -> None:
    assert SMALL + TINY == TokenUsage(input_tokens=105, output_tokens=11)


def test_the_sum_skips_steps_that_reported_nothing() -> None:
    assert TokenUsage.sum_reported((None, SMALL, None, TINY)) == SMALL + TINY


def test_a_run_that_reported_nothing_is_unmeasured_not_free() -> None:
    assert TokenUsage.sum_reported((None, None)) is None
    assert TokenUsage.sum_reported(()) is None


def test_a_call_usage_is_read_from_the_message_metadata() -> None:
    reported = UsageMetadata(input_tokens=100, output_tokens=10, total_tokens=110)
    assert TokenUsage.from_metadata(reported) == SMALL
