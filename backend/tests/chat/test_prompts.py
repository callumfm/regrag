"""Context formatting: numbered blocks the citation markers bind to."""

from app.chat.prompts import SYSTEM_PROMPT, build_gather_message, build_user_message, format_context
from app.ingestion.chunk.models import Reference
from tests.conftest import retrieved_chunk, search_result


def test_context_blocks_are_numbered_from_one():
    sources = (
        retrieved_chunk(),
        retrieved_chunk(id=2, celex="32015R0757", citation="Article 5"),
    )
    context = format_context(sources)
    assert context.startswith("[1] (32023R1805, Article 4(1))\n")
    assert "\n\n[2] (32015R0757, Article 5)\n" in context


def test_context_blocks_carry_the_chunk_text():
    context = format_context((retrieved_chunk(text="A very specific clause."),))
    assert "A very specific clause." in context


def test_user_message_puts_context_before_the_question():
    message = build_user_message("What is the limit?", (retrieved_chunk(),))
    assert message.index("[1]") < message.index("Question: What is the limit?")


def test_system_prompt_demands_inline_markers():
    assert "[1]" in SYSTEM_PROMPT


class TestBuildGatherMessage:
    def test_carries_numbered_blocks_and_the_question(self):
        sources = (search_result(text="A very specific clause."),)

        message = build_gather_message("What is the limit?", sources)

        assert "[1] (32023R1805, Article 4(1))" in message
        assert "A very specific clause." in message
        assert message.endswith("Question: What is the limit?")

    def test_lists_a_blocks_followable_references_with_their_addresses(self):
        reference = Reference(raw="Article 6(2)", article="6", paragraph="2")
        sources = (search_result(references=(reference,)),)

        message = build_gather_message("q", sources)

        assert "cites: 32023R1805 Article 6(2)" in message

    def test_names_the_cited_act_when_the_reference_crosses_acts(self):
        reference = Reference(raw="Article 3 of Regulation X", instrument="32015R0757", article="3")
        sources = (search_result(references=(reference,)),)

        message = build_gather_message("q", sources)

        assert "cites: 32015R0757 Article 3" in message

    def test_skips_references_that_name_no_division(self):
        reference = Reference(raw="Regulation (EU) 2015/757", instrument="32015R0757")
        sources = (search_result(references=(reference,)),)

        message = build_gather_message("q", sources)

        assert "cites:" not in message
