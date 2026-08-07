"""Blocks flattened to lines, and lines folded under the sub-headings that introduce them."""

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.consolidated import SUBHEADING_LEVEL_RE
from app.ingestion.parse.html.lines import Subheading, collect_lines, nest_under_subheadings
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument
from tests.ingestion.parse.html.helpers import (
    all_sections,
    annexes,
    articles,
    of_kind,
    subdivision,
)


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


def test_the_level_one_line_is_not_collected_because_it_is_the_annex_title():
    node = subdivision(
        '<html><body><div id="anx_I">'
        '<p class="title-gr-seq-level-1">Monitoring methods</p>'
        '<p class="title-gr-seq-level-2">A. First part</p>'
        '<p class="norm">Prose under A.</p>'
        "</div></body></html>",
        "anx_I",
    )
    assert collect_lines(node, SUBHEADING_LEVEL_RE) == [
        Subheading(2, "A. First part"),
        "Prose under A.",
    ]


def test_a_stream_with_no_subheadings_becomes_one_flat_paragraph():
    sections = nest_under_subheadings(["one", "two"])
    assert len(sections) == 1
    assert sections[0].kind is SectionKind.PARAGRAPH
    assert sections[0].text == "one\ntwo"


def test_layout_tables_flatten_into_the_paragraph_that_contains_them(fueleu: ParsedDocument):
    paragraph_2 = articles(fueleu)[0].children[1]
    assert paragraph_2.children == ()
    assert "2 % from 1 January 2025;" in paragraph_2.text
    assert "6 % from 1 January 2030;" in paragraph_2.text


def test_consolidated_grid_lists_flatten_into_the_paragraph(mrv: ParsedDocument):
    definitions = articles(mrv)[0]
    text = " ".join(p.text for p in definitions.children)
    assert "(a) ‘greenhouse gas emissions’ means" in text
    assert "(c) ‘voyage’ means" in text


def test_consolidated_annex_headings_nest_by_level(mrv: ParsedDocument):
    annex = annexes(mrv)[0]
    top = of_kind(annex.children, SectionKind.HEADING)
    assert top
    assert (top[0].title or "").startswith("A.")
    nested = of_kind(top[0].children, SectionKind.HEADING)
    assert nested
    assert (nested[0].title or "").startswith("1.")


def test_oj_annexes_are_flat(fueleu: ParsedDocument):
    annex = annexes(fueleu)[0]
    assert not of_kind(annex.children, SectionKind.HEADING)
    assert of_kind(annex.children, SectionKind.TABLE)


def test_consolidated_annex_prose_excludes_its_heading_lines(mrv: ParsedDocument):
    annex = annexes(mrv)[0]
    sections = list(all_sections(annex.children))
    prose = "\n".join(s.text for s in sections)
    titles = [s.title for s in sections if s.kind is SectionKind.HEADING and s.title]
    assert titles
    assert annex.title is not None
    assert annex.title not in prose
    assert not [title for title in titles if title in prose]


def test_consolidated_annex_prose_sits_under_the_heading_that_introduces_it(mrv: ParsedDocument):
    annex = annexes(mrv)[0]
    part_a = of_kind(annex.children, SectionKind.HEADING)[0]
    formulae = of_kind(part_a.children, SectionKind.HEADING)[0]
    prose = of_kind(formulae.children, SectionKind.PARAGRAPH)
    assert prose
    assert "companies shall apply the following formula" in prose[0].text


def test_consolidated_annex_prose_before_the_first_heading_stays_at_annex_level():
    document = parse_eurlex_html(
        "<html><body>"
        '<div class="eli-subdivision" id="art_1">'
        '<p class="title-article-norm">Article 1</p><p class="norm">Subject matter.</p></div>'
        '<div id="anx_I"><p class="title-annex-1">ANNEX I</p>'
        '<p class="title-gr-seq-level-1">Monitoring methods</p>'
        '<p class="norm">Preamble prose.</p>'
        '<p class="title-gr-seq-level-2">A. First part</p>'
        '<p class="norm">Prose under A.</p></div>'
        "</body></html>",
        "x",
        "t",
    )
    preamble, heading = annexes(document)[0].children
    assert preamble.kind is SectionKind.PARAGRAPH
    assert preamble.text == "Preamble prose."
    assert heading.kind is SectionKind.HEADING
    assert heading.title == "A. First part"
    assert [c.text for c in heading.children] == ["Prose under A."]
