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
