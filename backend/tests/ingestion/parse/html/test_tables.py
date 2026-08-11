"""Detaching data tables: the rows they yield, and the prose they must not leave behind."""

import pytest
from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.consolidated import CONSOLIDATED
from app.ingestion.parse.html.oj import OJ
from app.ingestion.parse.html.tables import detach_data_tables
from app.ingestion.parse.models import ParsedDocument
from tests.conftest import FUELEU_HTML, MRV_HTML
from tests.ingestion.parse.html.helpers import all_sections, annexes, subdivision


@pytest.fixture
def fueleu_annex() -> Node:
    """Function-scoped: detaching tables mutates the tree, so it cannot be shared."""
    return subdivision(FUELEU_HTML, "anx_II")


def test_data_table_rows_are_a_raw_grid(fueleu_annex: Node):
    grids = detach_data_tables(fueleu_annex, OJ)
    assert grids
    assert grids[0].kind is SectionKind.TABLE
    rows = grids[0].rows
    assert rows[0] == ("1", "2", "3", "4", "5", "6", "7", "8", "9")
    assert any("Fuel Class" in cell for row in rows for cell in row)


def test_extracted_rows_are_tuples_of_strings(fueleu_annex: Node):
    for grid in detach_data_tables(fueleu_annex, OJ):
        assert isinstance(grid.rows, tuple)
        for row in grid.rows:
            assert isinstance(row, tuple)
            assert all(isinstance(cell, str) for cell in row)


def test_formula_images_become_placeholders_in_table_cells(fueleu_annex: Node):
    cells = [
        cell for grid in detach_data_tables(fueleu_annex, OJ) for row in grid.rows for cell in row
    ]
    assert any("[formula]" in cell for cell in cells)
    assert not any("base64" in cell for cell in cells)


def test_extracting_a_table_detaches_it_so_its_text_is_not_duplicated(fueleu_annex: Node):
    assert "Fuel Class" in fueleu_annex.text()
    detach_data_tables(fueleu_annex, OJ)
    assert "Fuel Class" not in fueleu_annex.text()


def test_layout_tables_are_not_extracted_as_data_tables():
    article = subdivision(FUELEU_HTML, "art_4")
    assert article.css("table")
    assert detach_data_tables(article, OJ) == ()


def test_consolidated_data_table_rows_are_a_raw_grid():
    grids = detach_data_tables(subdivision(MRV_HTML, "anx_I"), CONSOLIDATED)
    assert len(grids) == 2
    assert grids[0].kind is SectionKind.TABLE
    assert grids[0].rows[0] == ("Term", "Explanation")


def test_consolidated_data_tables_do_not_stay_behind_as_annex_prose(mrv: ParsedDocument):
    annex = annexes(mrv)[0]
    prose = "\n".join(s.text or "" for s in all_sections(annex.children))
    assert "Explanation" not in prose
