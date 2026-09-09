"""Context formatting: numbered blocks the citation markers bind to."""

from app.chat.prompts import (
    ASSESS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_assess_message,
    build_assess_system_prompt,
    build_user_message,
    format_context,
    strip_markers,
)
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


class TestBuildAssessMessage:
    def test_carries_numbered_blocks_and_the_question(self):
        sources = (search_result(text="A very specific clause."),)

        message = build_assess_message("What is the limit?", sources)

        assert "[1] (32023R1805, Article 4(1))" in message
        assert "A very specific clause." in message
        assert message.endswith("Question: What is the limit?")

    def test_lists_a_blocks_followable_references_with_their_addresses(self):
        reference = Reference(raw="Article 6(2)", article="6", paragraph="2")
        sources = (search_result(references=(reference,)),)

        message = build_assess_message("q", sources)

        assert "cites: 32023R1805 Article 6(2)" in message

    def test_names_the_cited_act_when_the_reference_crosses_acts(self):
        reference = Reference(raw="Article 3 of Regulation X", instrument="32015R0757", article="3")
        sources = (search_result(references=(reference,)),)

        message = build_assess_message("q", sources)

        assert "cites: 32015R0757 Article 3" in message

    def test_names_an_address_once_however_many_points_of_it_are_cited(self):
        references = tuple(
            Reference(
                raw=f"Article 3, point ({point}), of Regulation X",
                instrument="32015R0757",
                article="3",
            )
            for point in "cen"
        )
        sources = (search_result(references=references),)

        message = build_assess_message("q", sources)

        assert message.count("32015R0757 Article 3") == 1

    def test_skips_references_that_name_no_division(self):
        reference = Reference(raw="Regulation (EU) 2015/757", instrument="32015R0757")
        sources = (search_result(references=(reference,)),)

        message = build_assess_message("q", sources)

        assert "cites:" not in message


class TestBuildAssessSystemPrompt:
    def test_with_refusal_allowed_the_prompt_adds_when_to_call_insufficient_context(self):
        prompt = build_assess_system_prompt(may_refuse=True)
        assert prompt.startswith(ASSESS_SYSTEM_PROMPT)
        assert "insufficient_context" in prompt[len(ASSESS_SYSTEM_PROMPT) :]

    def test_without_it_the_prompt_is_the_bare_one(self):
        assert build_assess_system_prompt(may_refuse=False) == ASSESS_SYSTEM_PROMPT


def test_strip_markers_removes_every_citation_marker_and_nothing_else():
    assert strip_markers("Ships must report.[1] Yearly.[2][3] Done.") == (
        "Ships must report. Yearly. Done."
    )
    assert strip_markers("No markers here.") == "No markers here."
