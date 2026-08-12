"""What the embed stage reports: its buckets, and how a failure is recorded against a document."""

from app.core.llm import LLMError
from app.ingestion.embed.models import EmbedOutcome


def test_a_fresh_outcome_is_empty():
    outcome = EmbedOutcome()
    assert (outcome.embedded, outcome.already_embedded, outcome.failed) == (0, 0, {})


def test_a_failure_is_recorded_under_its_document_with_its_type():
    outcome = EmbedOutcome()
    outcome.fail("32023R1805", LLMError("embedding call failed"))
    assert outcome.failed == {"32023R1805": "LLMError: embedding call failed"}
