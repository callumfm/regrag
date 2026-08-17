"""Context formatting and budget: numbered blocks the citation markers bind to."""

from app.chat.prompts import SYSTEM_PROMPT, build_user_message, fit_context, format_context
from tests.conftest import retrieved_chunk

ARTICLE_4 = (
    retrieved_chunk(id=1, text="Article 4 paragraph 1."),
    retrieved_chunk(id=2, citation="Article 4(2)", position=2, text="Article 4 paragraph 2."),
)
ARTICLE_5 = (
    retrieved_chunk(id=3, article="5", citation="Article 5(1)", position=3, text="Article 5."),
)


def chars(*sections: tuple) -> int:
    return sum(len(chunk.text) for section in sections for chunk in section)


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


def test_fit_context_keeps_everything_that_fits():
    assert fit_context(ARTICLE_4 + ARTICLE_5, chars(ARTICLE_4, ARTICLE_5)) == ARTICLE_4 + ARTICLE_5


def test_fit_context_drops_whole_sections_from_the_least_relevant_end():
    """One character short of both articles loses all of the second, not its tail."""
    assert fit_context(ARTICLE_4 + ARTICLE_5, chars(ARTICLE_4, ARTICLE_5) - 1) == ARTICLE_4


def test_fit_context_never_splits_a_section():
    """A budget that fits article 4's first paragraph but not its second drops neither
    paragraph: the section arrives whole because it is first, and article 5 not at all."""
    assert fit_context(ARTICLE_4 + ARTICLE_5, len(ARTICLE_4[0].text)) == ARTICLE_4


def test_fit_context_always_keeps_the_top_section():
    assert fit_context(ARTICLE_4 + ARTICLE_5, 1) == ARTICLE_4


def test_fit_context_of_nothing_is_nothing():
    assert fit_context((), 100) == ()
