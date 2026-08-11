"""The original Official Journal dialect: oj-* classes and dotted paragraph container ids."""

import re

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.dialect import Dialect
from app.ingestion.parse.html.lines import block_text
from app.ingestion.parse.models import Section
from app.ingestion.parse.text import LEADING_NUMBER_RE, clean_text

SIGNATURE = ".oj-normal"
ARTICLE_HEADING = "p.oj-ti-art"
PARAGRAPH_CONTAINER = "div[id]"
ANNEX_LABEL = "p.oj-doc-ti"
DATA_TABLE = "table.oj-table"
PARAGRAPH_ID_RE = re.compile(r"^\d+\.\d+$")


def paragraph_nodes(node: Node) -> list[Node]:
    """Paragraph containers have ids like 004.001; Node.css matches self, so exclude it."""
    own_id = node.id
    return [
        child
        for child in node.css(PARAGRAPH_CONTAINER)
        if child.id != own_id and PARAGRAPH_ID_RE.match(child.id or "")
    ]


def paragraph_section(node: Node) -> Section:
    """OJ paragraphs carry their number as a leading 'N.' in the text, which is split off."""
    number, text = None, block_text(node)
    match = LEADING_NUMBER_RE.match(text)
    if match:
        number, text = match.group(1), text[match.end() :]
    return Section(kind=SectionKind.PARAGRAPH, number=number, text=text)


def annex_title(node: Node) -> str | None:
    """OJ annexes repeat the same class for the label and the title that follows it."""
    labels = node.css(ANNEX_LABEL)
    return clean_text(labels[1].text()) if len(labels) > 1 else None


OJ = Dialect(
    signature=SIGNATURE,
    article_heading=ARTICLE_HEADING,
    annex_label=ANNEX_LABEL,
    data_table=DATA_TABLE,
    annex_title=annex_title,
    paragraph_nodes=paragraph_nodes,
    paragraph_section=paragraph_section,
    annex_subheading_level=None,
)
