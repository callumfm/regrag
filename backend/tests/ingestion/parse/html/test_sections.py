"""Articles, their numbered paragraphs, and annex labels and titles."""

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument
from tests.ingestion.parse.html.helpers import annexes, articles, of_kind


def test_oj_article_boundaries(fueleu: ParsedDocument):
    assert [a.number for a in articles(fueleu)] == ["4", "5"]


def test_consolidated_article_boundaries(mrv: ParsedDocument):
    assert [a.number for a in articles(mrv)] == ["3", "4", "11a"]


def test_oj_article_title(fueleu: ParsedDocument):
    assert articles(fueleu)[0].title == "GHG intensity limit on energy used on board by a ship"


def test_consolidated_article_title(mrv: ParsedDocument):
    assert articles(mrv)[0].title == "Definitions"


def test_lettered_article_title(mrv: ParsedDocument):
    lettered = articles(mrv)[2]
    assert lettered.number == "11a"
    assert lettered.title == (
        "Reporting and submission of the aggregated emissions data at company level"
    )


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


def test_oj_paragraphs_are_numbered_and_split_from_their_text(fueleu: ParsedDocument):
    first = articles(fueleu)[0]
    numbers = [p.number for p in first.children]
    assert numbers == ["1", "2", "3", "4"]
    assert first.children[0].text == (
        "The yearly average GHG intensity of the energy used on board by a ship "
        "during a reporting period shall not exceed the limit set out in paragraph 2."
    )


def test_consolidated_paragraphs_are_numbered(mrv: ParsedDocument):
    article_4 = articles(mrv)[1]
    assert [p.number for p in article_4.children] == ["1", "2", "3", "4", "5", "6", "7", "8"]


def test_consolidated_paragraph_text_excludes_its_number(mrv: ParsedDocument):
    article_4 = articles(mrv)[1]
    assert article_4.children[0].text.startswith("In accordance with Articles 8 to 12")


def test_amendment_markers_are_stripped_but_wrapped_text_survives(mrv: ParsedDocument):
    article_4 = articles(mrv)[1]
    text = article_4.children[1].text
    assert "►" not in text and "◄" not in text and "▼" not in text
    assert "greenhouse gas" in text


def test_oj_annex_label_and_title(fueleu: ParsedDocument):
    annex = annexes(fueleu)[0]
    assert annex.number == "II"
    assert annex.title == "Default emission factors"


def test_consolidated_annex_label_and_title(mrv: ParsedDocument):
    annex = annexes(mrv)[0]
    assert annex.number == "I"
    assert annex.title == "Methods for monitoring greenhouse gas emissions"


def test_an_annex_holding_no_blocks_still_yields_its_prose():
    document = parse_eurlex_html(
        "<html><body>"
        '<div class="eli-subdivision" id="art_1">'
        '<p class="oj-ti-art">Article 1</p><p class="oj-normal">Subject matter.</p></div>'
        '<div id="anx_I"><div class="oj-normal">Annex prose in a bare div.</div></div>'
        "</body></html>",
        "x",
        "t",
    )
    prose = of_kind(annexes(document)[0].children, SectionKind.PARAGRAPH)
    assert [section.text for section in prose] == ["Annex prose in a bare div."]


def test_oj_annex_prose_is_kept_alongside_its_tables(fueleu: ParsedDocument):
    annex = annexes(fueleu)[0]
    prose = of_kind(annex.children, SectionKind.PARAGRAPH)
    assert prose
    assert "The default emission factors contained in the table below" in prose[0].text
    assert "Fuel Class" not in prose[0].text
