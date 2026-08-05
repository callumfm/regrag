"""Format-neutral parser IR: section nesting and document identity."""

import pytest
from pydantic import ValidationError

from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import ParsedDocument, Section


def test_section_rejects_a_kind_that_is_not_a_section_kind():
    with pytest.raises(ValidationError):
        Section(kind="footnote")


def test_section_rejects_a_child_that_is_not_a_section():
    with pytest.raises(ValidationError):
        Section(kind=SectionKind.ARTICLE, children=("not a section",))  # ty: ignore[invalid-argument-type]


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
