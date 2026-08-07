"""Articles and annexes as Section subtrees, built the same way for either dialect."""

import re

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.dialect import ARTICLE_TITLE, Dialect
from app.ingestion.parse.html.lines import (
    Line,
    Subheading,
    collect_lines,
    nest_under_subheadings,
    prose_lines,
)
from app.ingestion.parse.html.tables import detach_data_tables
from app.ingestion.parse.models import Section
from app.ingestion.parse.text import ANNEX_NUMBER_RE, ARTICLE_NUMBER_RE, clean_text


def number_from_heading(node: Node, selector: str, pattern: re.Pattern[str]) -> str | None:
    heading = node.css_first(selector)
    if heading is None:
        return None
    match = pattern.search(clean_text(heading.text()))
    return match.group(1) if match else None


def build_paragraphs(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """Numbered paragraphs, falling back to one unnumbered paragraph for the whole article."""
    sections = [dialect.paragraph_section(child) for child in dialect.paragraph_nodes(node)]
    if sections:
        return tuple(sections)
    lines = prose_lines(node, dialect.article_heading, ARTICLE_TITLE)
    return (Section(kind=SectionKind.PARAGRAPH, text="\n".join(lines)),) if lines else ()


def build_article(node: Node, dialect: Dialect) -> Section:
    title = node.css_first(ARTICLE_TITLE)
    return Section(
        kind=SectionKind.ARTICLE,
        number=number_from_heading(node, dialect.article_heading, ARTICLE_NUMBER_RE),
        title=clean_text(title.text()) if title else None,
        children=build_paragraphs(node, dialect),
    )


def build_annex_body(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """The annex prose nested under its own sub-headings, with its label lines removed."""
    lines: list[Line] = []
    collect_lines(node, lines, dialect.annex_subheading)
    skip = {clean_text(label.text()) for label in node.css(dialect.annex_label)}
    return nest_under_subheadings(
        line for line in lines if isinstance(line, Subheading) or line not in skip
    )


def build_annex(node: Node, dialect: Dialect) -> Section:
    """OJ annexes are flat; consolidated ones nest by title-gr-seq level."""
    tables = detach_data_tables(node, dialect)
    return Section(
        kind=SectionKind.ANNEX,
        number=number_from_heading(node, dialect.annex_label, ANNEX_NUMBER_RE),
        title=dialect.annex_title(node),
        children=tables + build_annex_body(node, dialect),
    )
