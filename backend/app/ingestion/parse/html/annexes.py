"""Annexes as Section subtrees: heading and data tables detached, then the remaining
prose folded under the sub-headings that introduce it."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.dialect import Dialect
from app.ingestion.parse.html.paragraphs import CELL, Line, collect_lines, detach_texts
from app.ingestion.parse.html.text import ANNEX_NUMBER_RE, clean_text, heading_number
from app.ingestion.parse.models import Section

ANNEX_CONTAINER = "div[id^=anx_]"


def _table_rows(node: Node) -> tuple[tuple[str, ...], ...]:
    """A table as a raw grid of cleaned cells, with no row treated as a header."""
    rows = []
    for row in node.css("tr"):
        cells = tuple(clean_text(cell.text()) for cell in row.css(CELL))
        if cells:
            rows.append(cells)
    return tuple(rows)


def _detach_data_tables(node: Node, selector: str) -> tuple[Section, ...]:
    """Data tables as TABLE sections, removed from the tree so their cells are not re-read."""
    sections = []
    for table in node.css(selector):
        if rows := _table_rows(table):
            sections.append(Section(kind=SectionKind.TABLE, rows=rows))
        table.replace_with("")
    return tuple(sections)


@dataclass
class _OpenSubheading:
    """A sub-heading still being filled: its own prose, then its subsections."""

    level: int
    title: str | None
    lines: list[str] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)

    def flush_prose(self) -> None:
        """Bank the prose read so far as a paragraph, ahead of any subsection after it."""
        if self.lines:
            self.children.append(Section(kind=SectionKind.PARAGRAPH, text="\n".join(self.lines)))
            self.lines.clear()

    def close(self) -> Section:
        """Bank any prose still open and return the finished heading section."""
        self.flush_prose()
        return Section(kind=SectionKind.HEADING, title=self.title, children=tuple(self.children))


def _nest_under_subheadings(lines: Iterable[Line]) -> tuple[Section, ...]:
    """Fold a line stream into sections, each sub-heading owning the prose beneath it."""
    stack = [_OpenSubheading(level=0, title=None)]

    def unwind(level: int) -> None:
        while stack[-1].level >= level:
            done = stack.pop()
            stack[-1].children.append(done.close())

    for line in lines:
        if isinstance(line, str):
            stack[-1].lines.append(line)
        else:
            unwind(line.level)
            stack[-1].flush_prose()
            stack.append(_OpenSubheading(level=line.level, title=line.title))
    unwind(1)
    stack[0].flush_prose()
    return tuple(stack[0].children)


def build_annex(node: Node, dialect: Dialect) -> Section:
    """An annex as a Section: detach the heading, detach the data tables, and what
    remains is body prose; OJ annexes are flat, consolidated ones nest by level.
    """
    labels = detach_texts(node, dialect.annex_label)
    titles = detach_texts(node, dialect.annex_title) if dialect.annex_title else labels[1:]
    tables = _detach_data_tables(node, dialect.data_table)
    body = _nest_under_subheadings(collect_lines(node, dialect.subheading_re))
    return Section(
        kind=SectionKind.ANNEX,
        number=heading_number(labels, ANNEX_NUMBER_RE),
        title=titles[0] if titles else None,
        children=tables + body,
    )
