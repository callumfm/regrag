"""Whole-document parsing: dialect detection, section order, and stripped markup."""

import pytest
from selectolax.parser import HTMLParser

from app.ingestion.enums import SectionKind
from app.ingestion.exceptions import ParseError
from app.ingestion.parse.html.document import parse_eurlex_html
from app.ingestion.parse.html.text import FOOTNOTE_BLOCK, drop_non_legal_markup
from app.ingestion.parse.models import ParsedDocument
from tests.conftest import FUELEU_HTML, MRV_HTML
from tests.ingestion.parse.html.helpers import all_sections, articles


def test_unrecognised_dialect_raises():
    with pytest.raises(ParseError, match="dialect"):
        parse_eurlex_html("<html><body><p>hello</p></body></html>")


def test_document_without_articles_raises():
    with pytest.raises(ParseError, match="no articles"):
        parse_eurlex_html('<html><body><p class="oj-normal">prose</p></body></html>')


def test_annexes_follow_articles_in_document_order(fueleu: ParsedDocument):
    kinds = [s.kind for s in fueleu.sections]
    assert kinds == [SectionKind.ARTICLE, SectionKind.ARTICLE, SectionKind.ANNEX]


def test_recitals_and_citations_are_excluded():
    sections = parse_eurlex_html(
        "<html><body>"
        '<div class="eli-subdivision" id="rct_1"><p class="oj-normal">Whereas something.</p></div>'
        '<div class="eli-subdivision" id="cit_1">'
        '<p class="oj-normal">Having regard to the Treaty.</p></div>'
        '<div class="eli-subdivision" id="art_1"><p class="oj-ti-art">Article 1</p>'
        '<p class="oj-normal">Subject matter.</p></div>'
        "</body></html>"
    )
    text = " ".join(s.text for s in all_sections(sections))
    assert "Subject matter." in text
    assert "Whereas" not in text
    assert "Having regard to the Treaty" not in text


def test_footnote_blocks_are_removed_from_the_tree():
    for html in (MRV_HTML, FUELEU_HTML):
        tree = HTMLParser(html)
        drop_non_legal_markup(tree)
        assert tree.css(FOOTNOTE_BLOCK) == []


def test_footnote_superscripts_are_dropped_with_their_brackets(mrv: ParsedDocument):
    definitions = articles(mrv.sections)[0].children[0]
    assert "of the European Parliament and of the Council;" in definitions.text
    assert "( 1 )" not in definitions.text
