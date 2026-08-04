"""EUR-Lex HTML parser: article boundaries, paragraphs, tables and annexes."""

from pathlib import Path

import pytest

from app.ingestion.enums import SectionKind
from app.ingestion.parse.base import ParseError
from app.ingestion.parse.eurlex_html import parse_eurlex_html

FIXTURES = Path(__file__).parent / "fixtures"
FUELEU = (FIXTURES / "32023R1805.html").read_text()
MRV = (FIXTURES / "32015R0757.html").read_text()


def fueleu():
    return parse_eurlex_html(FUELEU, "32023R1805", "fueleu")


def mrv():
    return parse_eurlex_html(MRV, "32015R0757", "mrv")


def articles(document):
    return [s for s in document.sections if s.kind is SectionKind.ARTICLE]


def test_carries_identity_from_the_ingest_record():
    document = fueleu()
    assert document.ref == "32023R1805"
    assert document.topic == "fueleu"


def test_oj_article_boundaries():
    assert [a.number for a in articles(fueleu())] == ["4", "5"]


def test_consolidated_article_boundaries():
    assert [a.number for a in articles(mrv())] == ["3", "4", "11a"]


def test_oj_article_title():
    assert articles(fueleu())[0].title == "GHG intensity limit on energy used on board by a ship"


def test_consolidated_article_title():
    assert articles(mrv())[0].title == "Definitions"


def test_lettered_article_title():
    lettered = articles(mrv())[2]
    assert lettered.number == "11a"
    assert lettered.title == (
        "Reporting and submission of the aggregated emissions data at company level"
    )


def test_unrecognised_dialect_raises():
    with pytest.raises(ParseError, match="dialect"):
        parse_eurlex_html("<html><body><p>hello</p></body></html>", "x", "t")


def test_document_without_articles_raises():
    with pytest.raises(ParseError, match="no articles"):
        parse_eurlex_html('<html><body><p class="oj-normal">prose</p></body></html>', "x", "t")


def test_oj_paragraphs_are_numbered_and_split_from_their_text():
    first = articles(fueleu())[0]
    numbers = [p.number for p in first.children]
    assert numbers == ["1", "2", "3", "4"]
    assert first.children[0].text == (
        "The yearly average GHG intensity of the energy used on board by a ship "
        "during a reporting period shall not exceed the limit set out in paragraph 2."
    )


def test_consolidated_paragraphs_are_numbered():
    article_4 = articles(mrv())[1]
    assert [p.number for p in article_4.children] == ["1", "2", "3", "4", "5", "6", "7", "8"]


def test_consolidated_paragraph_text_excludes_its_number():
    article_4 = articles(mrv())[1]
    assert article_4.children[0].text.startswith("In accordance with Articles 8 to 12")


def test_amendment_markers_are_stripped_but_wrapped_text_survives():
    article_4 = articles(mrv())[1]
    text = article_4.children[1].text
    assert "►" not in text and "◄" not in text and "▼" not in text
    assert "greenhouse gas" in text


def test_layout_tables_flatten_into_the_paragraph_that_contains_them():
    paragraph_2 = articles(fueleu())[0].children[1]
    assert paragraph_2.children == ()
    assert "2 % from 1 January 2025;" in paragraph_2.text
    assert "6 % from 1 January 2030;" in paragraph_2.text


def test_footnote_superscripts_are_dropped_with_their_brackets():
    definitions = articles(mrv())[0].children[0]
    assert "of the European Parliament and of the Council;" in definitions.text
    assert "( 1 )" not in definitions.text


def test_consolidated_grid_lists_flatten_into_the_paragraph():
    definitions = articles(mrv())[0]
    text = " ".join(p.text for p in definitions.children)
    assert "(a) ‘greenhouse gas emissions’ means" in text
    assert "(c) ‘voyage’ means" in text


def test_article_without_numbered_paragraphs_yields_one_unnumbered_paragraph():
    document = parse_eurlex_html(
        '<html><body><div class="eli-subdivision" id="art_1">'
        '<p class="oj-ti-art">Article 1</p>'
        '<p class="oj-normal">This Regulation enters into force.</p>'
        "</div></body></html>",
        "x",
        "t",
    )
    paragraphs = document.sections[0].children
    assert len(paragraphs) == 1
    assert paragraphs[0].number is None
    assert paragraphs[0].text == "This Regulation enters into force."
