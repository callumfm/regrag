"""The two EUR-Lex markup dialects: the shape of what they disagree about, one
instance per dialect, and how to tell which one a document is written in."""

import re
from collections.abc import Callable
from dataclasses import dataclass

from selectolax.parser import HTMLParser, Node

from app.ingestion.exceptions import ParseError
from app.ingestion.parse.html import consolidated, oj
from app.ingestion.parse.models import Section


@dataclass(frozen=True)
class Dialect:
    """One EUR-Lex markup dialect: how to recognise it and where it keeps each part."""

    signature: str
    article_heading: str
    annex_label: str
    annex_title: str | None  # None: the title is the annex_label match after the label
    data_table: str
    subheading_re: re.Pattern[str] | None
    find_paragraphs: Callable[[Node], list[Node]]
    build_paragraph: Callable[[Node], Section]


OJ = Dialect(
    signature=oj.SIGNATURE,
    article_heading=oj.ARTICLE_HEADING,
    annex_label=oj.ANNEX_LABEL,
    annex_title=None,
    data_table=oj.DATA_TABLE,
    subheading_re=None,
    find_paragraphs=oj.find_paragraphs,
    build_paragraph=oj.build_paragraph,
)

CONSOLIDATED = Dialect(
    signature=consolidated.SIGNATURE,
    article_heading=consolidated.ARTICLE_HEADING,
    annex_label=consolidated.ANNEX_LABEL,
    annex_title=consolidated.ANNEX_TITLE,
    data_table=consolidated.DATA_TABLE,
    subheading_re=consolidated.SUBHEADING_LEVEL_RE,
    find_paragraphs=consolidated.find_paragraphs,
    build_paragraph=consolidated.build_paragraph,
)

DIALECTS = (OJ, CONSOLIDATED)


def detect_dialect(tree: HTMLParser) -> Dialect:
    """OJ documents carry oj-* classes; consolidated ones carry norm."""
    for dialect in DIALECTS:
        if tree.css_first(dialect.signature) is not None:
            return dialect
    raise ParseError("unrecognised EUR-Lex dialect")
