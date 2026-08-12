"""What the embed stage reports: its buckets, and how a failure is recorded against a document."""

from app.core.llm import LLMError
from app.ingestion.embed.models import EmbedOutcome


def test_a_fresh_outcome_is_empty():
    outcome = EmbedOutcome()
    assert (outcome.embedded, outcome.already_embedded, outcome.failed) == (0, 0, {})


def test_a_failure_is_recorded_under_its_document_with_its_type():
    outcome = EmbedOutcome()
    outcome.fail("32023R1805", LLMError("embedding call failed"), chunks=64)
    assert outcome.failures == {"32023R1805": "64 chunks: LLMError: embedding call failed"}


def test_a_single_chunk_failure_reads_in_the_singular():
    outcome = EmbedOutcome()
    outcome.fail("32023R1805", LLMError("boom"), chunks=1)
    assert outcome.failures == {"32023R1805": "1 chunk: LLMError: boom"}


def test_repeated_failures_accumulate_chunks_and_keep_the_first_reason():
    outcome = EmbedOutcome()
    outcome.fail("32023R1805", LLMError("rate limited"), chunks=64)
    outcome.fail("32023R1805", LLMError("timed out"), chunks=40)
    assert outcome.failures == {"32023R1805": "104 chunks: LLMError: rate limited"}
