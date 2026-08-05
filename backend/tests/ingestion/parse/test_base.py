"""Format-neutral IR: node construction and text normalisation."""

from app.ingestion.enums import SectionKind
from app.ingestion.parse.base import ParsedDocument, Section, normalise


def test_section_defaults_to_an_empty_leaf():
    section = Section(kind=SectionKind.PARAGRAPH)
    assert section.number is None
    assert section.title is None
    assert section.text == ""
    assert section.rows == ()
    assert section.children == ()


def test_sections_nest():
    child = Section(kind=SectionKind.PARAGRAPH, number="1", text="text")
    parent = Section(kind=SectionKind.ARTICLE, number="4", children=(child,))
    assert parent.children[0].number == "1"


def test_parsed_document_carries_identity_from_the_ingest_record():
    document = ParsedDocument(ref="32023R1805", topic="fueleu", sections=())
    assert document.ref == "32023R1805"
    assert document.topic == "fueleu"


def test_normalise_collapses_runs_of_whitespace():
    assert normalise("a  \n   b\t c") == "a b c"


def test_normalise_replaces_the_non_breaking_spaces_eurlex_indents_with():
    assert normalise("1.\xa0\xa0\xa0The yearly average") == "1. The yearly average"


def test_normalise_strips_leading_and_trailing_whitespace():
    assert normalise("\n   ANNEX I\n   ") == "ANNEX I"
