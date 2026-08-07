"""Folding a line stream into sections, each sub-heading owning the prose beneath it."""

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.lines import Subheading, nest_under_subheadings


def test_nesting_puts_prose_under_the_subheading_that_introduces_it():
    sections = nest_under_subheadings([Subheading(2, "A. First"), "prose under A"])
    assert len(sections) == 1
    assert sections[0].kind is SectionKind.HEADING
    assert sections[0].title == "A. First"
    assert [child.text for child in sections[0].children] == ["prose under A"]


def test_nesting_deepens_then_returns_for_a_shallower_subheading():
    sections = nest_under_subheadings(
        [
            Subheading(2, "A"),
            "under A",
            Subheading(3, "A.1"),
            "under A.1",
            Subheading(2, "B"),
            "under B",
        ]
    )
    assert [section.title for section in sections] == ["A", "B"]
    prose, nested = sections[0].children
    assert prose.kind is SectionKind.PARAGRAPH
    assert prose.text == "under A"
    assert nested.kind is SectionKind.HEADING
    assert nested.title == "A.1"
    assert [child.text for child in nested.children] == ["under A.1"]


def test_prose_before_the_first_subheading_stays_at_the_top_level():
    sections = nest_under_subheadings(["preamble", Subheading(2, "A"), "under A"])
    assert sections[0].kind is SectionKind.PARAGRAPH
    assert sections[0].text == "preamble"
    assert sections[1].kind is SectionKind.HEADING


def test_a_level_one_subheading_is_dropped_because_it_repeats_the_annex_title():
    sections = nest_under_subheadings([Subheading(1, "ANNEX I"), "prose"])
    assert len(sections) == 1
    assert sections[0].kind is SectionKind.PARAGRAPH
    assert sections[0].text == "prose"


def test_a_stream_with_no_subheadings_becomes_one_flat_paragraph():
    sections = nest_under_subheadings(["one", "two"])
    assert len(sections) == 1
    assert sections[0].kind is SectionKind.PARAGRAPH
    assert sections[0].text == "one\ntwo"
