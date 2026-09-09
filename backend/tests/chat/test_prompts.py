"""Context formatting: numbered blocks the citation markers bind to."""

from app.chat.prompts import format_context
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
