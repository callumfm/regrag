"""Context formatting: numbered blocks the citation markers bind to."""

from app.chat.prompts import SYSTEM_PROMPT, build_user_message, format_context
from tests.conftest import retrieved_chunk


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
