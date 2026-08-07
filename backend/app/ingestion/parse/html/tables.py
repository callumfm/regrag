"""Data tables leave the tree as row grids, so their cells never re-appear as prose."""

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.dialect import CELL, Dialect
from app.ingestion.parse.models import Section
from app.ingestion.parse.text import clean_text


def table_rows(node: Node) -> tuple[tuple[str, ...], ...]:
    rows = []
    for row in node.css("tr"):
        cells = tuple(clean_text(cell.text()) for cell in row.css(CELL))
        if cells:
            rows.append(cells)
    return tuple(rows)


def detach_data_tables(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """Take data tables out of the tree, so their cells never re-appear as prose."""
    tables = node.css(dialect.data_table)
    sections = tuple(
        Section(kind=SectionKind.TABLE, rows=rows)
        for table in tables
        if (rows := table_rows(table))
    )
    for table in tables:
        table.decompose()
    return sections
