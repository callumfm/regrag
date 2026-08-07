"""What the two EUR-Lex markup dialects share, and the shape of what they do not."""

from collections.abc import Callable
from dataclasses import dataclass

from selectolax.parser import Node

from app.ingestion.parse.models import Section

ARTICLE_CONTAINER = "div.eli-subdivision[id^=art_]"
ANNEX_CONTAINER = "div[id^=anx_]"
ARTICLE_TITLE = "div.eli-title p"
CELL = "td, th"


@dataclass(frozen=True)
class Dialect:
    """One EUR-Lex markup dialect: how to recognise it and where it keeps each part."""

    signature: str
    article_heading: str
    annex_label: str
    data_table: str
    annex_title: Callable[[Node], str | None]
    paragraph_nodes: Callable[[Node], list[Node]]
    paragraph_section: Callable[[Node], Section]
    annex_subheading: str | None = None
