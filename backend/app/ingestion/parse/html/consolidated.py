"""The consolidated dialect: norm classes, title-gr-seq-level-N sub-headings, change bars."""

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.dialect import Dialect
from app.ingestion.parse.html.lines import block_text
from app.ingestion.parse.models import Section
from app.ingestion.parse.text import LEADING_NUMBER_RE, clean_text

SIGNATURE = "p.norm, div.norm"
ARTICLE_HEADING = "p.title-article-norm"
PARAGRAPH_CONTAINER = "div.norm"
PARAGRAPH_NUMBER = "span.no-parag"
PARAGRAPH_TEXT = "div.norm.inline-element"
ANNEX_LABEL = "p.title-annex-1"
ANNEX_TITLE = "p.title-gr-seq-level-1"
ANNEX_SUBHEADING = 'p[class^="title-gr-seq-level-"]'
DATA_TABLE = "table.borderOj"


def paragraph_nodes(node: Node) -> list[Node]:
    return [child for child in node.css(PARAGRAPH_CONTAINER) if child.css_first(PARAGRAPH_NUMBER)]


def paragraph_section(node: Node) -> Section:
    marker = node.css_first(PARAGRAPH_NUMBER)
    body = node.css_first(PARAGRAPH_TEXT)
    number = LEADING_NUMBER_RE.match(clean_text(marker.text()) if marker else "")
    return Section(
        kind=SectionKind.PARAGRAPH,
        number=number.group(1) if number else None,
        text=block_text(body) if body is not None else "",
    )


def annex_title(node: Node) -> str | None:
    title = node.css_first(ANNEX_TITLE)
    return clean_text(title.text()) if title is not None else None


CONSOLIDATED = Dialect(
    signature=SIGNATURE,
    article_heading=ARTICLE_HEADING,
    annex_label=ANNEX_LABEL,
    data_table=DATA_TABLE,
    annex_title=annex_title,
    paragraph_nodes=paragraph_nodes,
    paragraph_section=paragraph_section,
    annex_subheading=ANNEX_SUBHEADING,
)
