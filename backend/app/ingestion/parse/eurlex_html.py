"""EUR-Lex HTML parser: one traversal over the OJ and consolidated dialects."""

import re

from selectolax.parser import HTMLParser, Node

from app.ingestion.enums import SectionKind
from app.ingestion.exceptions import ParseError
from app.ingestion.parse.html.consolidated import CONSOLIDATED
from app.ingestion.parse.html.dialect import (
    ANNEX_CONTAINER,
    ARTICLE_CONTAINER,
    ARTICLE_TITLE,
    Dialect,
)
from app.ingestion.parse.html.lines import (
    Line,
    Subheading,
    collect_lines,
    nest_under_subheadings,
    prose_lines,
)
from app.ingestion.parse.html.oj import OJ
from app.ingestion.parse.html.tables import detach_data_tables
from app.ingestion.parse.models import ParsedDocument, Section
from app.ingestion.parse.text import (
    ANNEX_NUMBER_RE,
    ARTICLE_NUMBER_RE,
    FORMULA_PLACEHOLDER,
    clean_text,
)

FOOTNOTE_REF = "span.superscript, span.oj-super"
FOOTNOTE = "p.footnote, p.oj-note, div[id^=fnp]"
DROP = ("p.modref", FOOTNOTE_REF, FOOTNOTE)


def prepare(html: str) -> HTMLParser:
    """Placeholder every base64 image and drop the non-prose furniture."""
    tree = HTMLParser(html)
    for image in tree.css("img"):
        if (image.attributes.get("src") or "").startswith("data:"):
            image.replace_with(FORMULA_PLACEHOLDER)
    for selector in DROP:
        for node in tree.css(selector):
            node.decompose()
    return tree


DIALECTS = (OJ, CONSOLIDATED)


def detect(tree: HTMLParser) -> Dialect:
    """OJ documents carry oj-* classes; consolidated ones carry norm."""
    for dialect in DIALECTS:
        if tree.css_first(dialect.signature) is not None:
            return dialect
    raise ParseError("unrecognised EUR-Lex dialect")


def paragraph_sections(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """Numbered paragraphs, falling back to one unnumbered paragraph for the whole article."""
    sections = [dialect.paragraph_section(child) for child in dialect.paragraph_nodes(node)]
    if sections:
        return tuple(sections)
    lines = prose_lines(node, dialect.article_heading, ARTICLE_TITLE)
    return (Section(kind=SectionKind.PARAGRAPH, text="\n".join(lines)),) if lines else ()


def heading_number(node: Node, selector: str, pattern: re.Pattern[str]) -> str | None:
    heading = node.css_first(selector)
    if heading is None:
        return None
    match = pattern.search(clean_text(heading.text()))
    return match.group(1) if match else None


def article_section(node: Node, dialect: Dialect) -> Section:
    title = node.css_first(ARTICLE_TITLE)
    return Section(
        kind=SectionKind.ARTICLE,
        number=heading_number(node, dialect.article_heading, ARTICLE_NUMBER_RE),
        title=clean_text(title.text()) if title else None,
        children=paragraph_sections(node, dialect),
    )


def annex_body(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """The annex prose nested under its own sub-headings, with its label lines removed."""
    lines: list[Line] = []
    collect_lines(node, lines, dialect.annex_subheading)
    skip = {clean_text(label.text()) for label in node.css(dialect.annex_label)}
    return nest_under_subheadings(
        line for line in lines if isinstance(line, Subheading) or line not in skip
    )


def annex_section(node: Node, dialect: Dialect) -> Section:
    """OJ annexes are flat; consolidated ones nest by title-gr-seq level."""
    tables = detach_data_tables(node, dialect)
    return Section(
        kind=SectionKind.ANNEX,
        number=heading_number(node, dialect.annex_label, ANNEX_NUMBER_RE),
        title=dialect.annex_title(node),
        children=tables + annex_body(node, dialect),
    )


def parse_eurlex_html(html: str, celex: str, topic: str) -> ParsedDocument:
    """Parse one EUR-Lex document into the format-neutral section tree."""
    tree = prepare(html)
    dialect = detect(tree)
    articles = [article_section(node, dialect) for node in tree.css(ARTICLE_CONTAINER)]
    if not articles:
        raise ParseError(f"{celex}: no articles found")
    annexes = [annex_section(node, dialect) for node in tree.css(ANNEX_CONTAINER)]
    return ParsedDocument(celex=celex, topic=topic, sections=tuple(articles + annexes))
