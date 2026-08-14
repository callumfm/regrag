"""Articles as Section subtrees: heading and title detached, paragraphs built beneath."""

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.dialect import Dialect
from app.ingestion.parse.html.paragraphs import block_text, detach_texts
from app.ingestion.parse.html.text import ARTICLE_NUMBER_RE, heading_number
from app.ingestion.parse.models import Section

ARTICLE_CONTAINER = "div.eli-subdivision[id^=art_]"
ARTICLE_TITLE = "div.eli-title p"


def _build_paragraphs(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """Numbered paragraphs, falling back to one unnumbered paragraph for the whole article."""
    numbered = [dialect.build_paragraph(child) for child in dialect.find_paragraphs(node)]
    if numbered:
        return tuple(numbered)
    text = block_text(node)
    return (Section(kind=SectionKind.PARAGRAPH, text=text),) if text else ()


def build_article(node: Node, dialect: Dialect) -> Section:
    """An article as a Section: its heading and title read then detached, so the
    paragraphs are built from a node holding only body text.
    """
    headings = detach_texts(node, dialect.article_heading)
    titles = detach_texts(node, ARTICLE_TITLE)
    number = heading_number(headings, ARTICLE_NUMBER_RE)
    title = titles[0] if titles else None
    paragraphs = _build_paragraphs(node, dialect)
    return Section(kind=SectionKind.ARTICLE, number=number, title=title, children=paragraphs)
