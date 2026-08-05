"""EUR-Lex HTML parser: article boundaries, paragraphs, tables and annexes."""

from pathlib import Path

import pytest

from app.ingestion.enums import SectionKind
from app.ingestion.parse.base import ParseError
from app.ingestion.parse.eurlex_html import (
    CONS,
    FOOTNOTE,
    OJ,
    extract_tables,
    parse_eurlex_html,
    prepare,
)

FIXTURES = Path(__file__).parent / "fixtures"
FUELEU = (FIXTURES / "32023R1805.html").read_text()
MRV = (FIXTURES / "32015R0757.html").read_text()


def fueleu():
    return parse_eurlex_html(FUELEU, "32023R1805", "fueleu")


def mrv():
    return parse_eurlex_html(MRV, "32015R0757", "mrv")


def of_kind(sections, kind):
    return [s for s in sections if s.kind is kind]


def articles(document):
    return of_kind(document.sections, SectionKind.ARTICLE)


def annexes(document):
    return of_kind(document.sections, SectionKind.ANNEX)


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


def subdivision(html, node_id):
    node = prepare(html).css_first(f'div[id="{node_id}"]')
    assert node is not None
    return node


def test_data_table_rows_are_a_raw_grid():
    grids = extract_tables(subdivision(FUELEU, "anx_II"), OJ)
    assert grids
    assert grids[0].kind is SectionKind.TABLE
    rows = grids[0].rows
    assert rows[0] == ("1", "2", "3", "4", "5", "6", "7", "8", "9")
    assert any("Fuel Class" in cell for row in rows for cell in row)


def test_extracted_rows_are_tuples_of_strings():
    for grid in extract_tables(subdivision(FUELEU, "anx_II"), OJ):
        assert isinstance(grid.rows, tuple)
        for row in grid.rows:
            assert isinstance(row, tuple)
            assert all(isinstance(cell, str) for cell in row)


def test_formula_images_become_placeholders_in_table_cells():
    cells = [
        cell
        for grid in extract_tables(subdivision(FUELEU, "anx_II"), OJ)
        for row in grid.rows
        for cell in row
    ]
    assert any("[formula]" in cell for cell in cells)
    assert not any("base64" in cell for cell in cells)


def test_extracting_a_table_detaches_it_so_its_text_is_not_duplicated():
    annex = subdivision(FUELEU, "anx_II")
    assert "Fuel Class" in annex.text()
    extract_tables(annex, OJ)
    assert "Fuel Class" not in annex.text()


def test_layout_tables_are_not_extracted_as_data_tables():
    article = subdivision(FUELEU, "art_4")
    assert article.css("table")
    assert extract_tables(article, OJ) == ()


def test_consolidated_data_table_rows_are_a_raw_grid():
    grids = extract_tables(subdivision(MRV, "anx_I"), CONS)
    assert len(grids) == 2
    assert grids[0].kind is SectionKind.TABLE
    assert grids[0].rows[0] == ("Term", "Explanation")


def test_consolidated_data_tables_do_not_stay_behind_as_annex_prose():
    annex = annexes(mrv())[0]
    prose = "\n".join(s.text or "" for s in all_sections(annex.children))
    assert "Explanation" not in prose


def all_sections(sections):
    for section in sections:
        yield section
        yield from all_sections(section.children)


def test_oj_annex_label_and_title():
    annex = annexes(fueleu())[0]
    assert annex.number == "II"
    assert annex.title == "Default emission factors"


def test_consolidated_annex_label_and_title():
    annex = annexes(mrv())[0]
    assert annex.number == "I"
    assert annex.title == "Methods for monitoring greenhouse gas emissions"


def test_consolidated_annex_headings_nest_by_level():
    annex = annexes(mrv())[0]
    top = of_kind(annex.children, SectionKind.HEADING)
    assert top
    assert top[0].title.startswith("A.")
    nested = of_kind(top[0].children, SectionKind.HEADING)
    assert nested
    assert nested[0].title.startswith("1.")


def test_oj_annexes_are_flat():
    annex = annexes(fueleu())[0]
    assert not of_kind(annex.children, SectionKind.HEADING)
    assert of_kind(annex.children, SectionKind.TABLE)


def test_annexes_follow_articles_in_document_order():
    kinds = [s.kind for s in fueleu().sections]
    assert kinds == [SectionKind.ARTICLE, SectionKind.ARTICLE, SectionKind.ANNEX]


def test_recitals_and_citations_are_excluded():
    document = parse_eurlex_html(
        "<html><body>"
        '<div class="eli-subdivision" id="rct_1"><p class="oj-normal">Whereas something.</p></div>'
        '<div class="eli-subdivision" id="cit_1">'
        '<p class="oj-normal">Having regard to the Treaty.</p></div>'
        '<div class="eli-subdivision" id="art_1"><p class="oj-ti-art">Article 1</p>'
        '<p class="oj-normal">Subject matter.</p></div>'
        "</body></html>",
        "x",
        "t",
    )
    text = " ".join(s.text for s in all_sections(document.sections))
    assert "Subject matter." in text
    assert "Whereas" not in text
    assert "Having regard to the Treaty" not in text


def test_footnote_blocks_are_removed_from_the_tree():
    assert prepare(MRV).css(FOOTNOTE) == []
    assert prepare(FUELEU).css(FOOTNOTE) == []


def test_oj_annex_prose_is_kept_alongside_its_tables():
    annex = annexes(fueleu())[0]
    prose = of_kind(annex.children, SectionKind.PARAGRAPH)
    assert prose
    assert "The default emission factors contained in the table below" in prose[0].text
    assert "Fuel Class" not in prose[0].text


def test_consolidated_annex_prose_excludes_its_heading_lines():
    annex = annexes(mrv())[0]
    prose = of_kind(annex.children, SectionKind.PARAGRAPH)
    assert prose
    assert "Methods for monitoring greenhouse gas emissions" not in prose[0].text
    assert "companies shall apply the following formula" in prose[0].text
