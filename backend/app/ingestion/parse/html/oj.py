"""The original Official Journal dialect: oj-* classes and dotted paragraph container ids."""

import re

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.paragraphs import block_text
from app.ingestion.parse.html.text import LEADING_NUMBER_RE
from app.ingestion.parse.models import Section

SIGNATURE = ".oj-normal"
ARTICLE_HEADING = "p.oj-ti-art"
PARAGRAPH_CONTAINER = "div[id]"
ANNEX_LABEL = "p.oj-doc-ti"
DATA_TABLE = "table.oj-table"
PARAGRAPH_ID_RE = re.compile(r"^\d+\.\d+$")


def find_paragraphs(node: Node) -> list[Node]:
    """Paragraph containers have ids like 004.001; Node.css matches self, so exclude it."""
    own_id = node.id
    return [
        child
        for child in node.css(PARAGRAPH_CONTAINER)
        if child.id != own_id and PARAGRAPH_ID_RE.match(child.id or "")
    ]


def build_paragraph(node: Node) -> Section:
    """OJ paragraphs carry their number as a leading 'N.' in the text, which is split off."""
    number, text = None, block_text(node)
    match = LEADING_NUMBER_RE.match(text)
    if match:
        number, text = match.group(1), text[match.end() :]
    return Section(kind=SectionKind.PARAGRAPH, number=number, text=text)
