"""EUR-Lex HTML parser: one traversal over the OJ and consolidated dialects."""

import re
from dataclasses import dataclass

from selectolax.parser import HTMLParser, Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.base import (
    FORMULA_PLACEHOLDER,
    ParsedDocument,
    ParseError,
    Section,
    normalise,
)

ARTICLE_NUMBER = re.compile(r"Article\s+(\d+[a-z]?)", re.IGNORECASE)
ANNEX_NUMBER = re.compile(r"ANNEX\s+([IVXLC]+|\d+)", re.IGNORECASE)
PARAGRAPH_ID = re.compile(r"^\d+\.\d+$")
LEADING_NUMBER = re.compile(r"^(\d+[a-z]?)\.\s*")
MARKERS = re.compile(r"[▼►]\s*[A-Z]+\d*|◄")

ARTICLE = "div.eli-subdivision[id^=art_]"
ANNEX = "div[id^=anx_]"
ARTICLE_TITLE = "div.eli-title p"

OJ_PARAGRAPH = "div[id]"
CONS_PARAGRAPH = "div.norm"
CONS_PARAGRAPH_NUMBER = "span.no-parag"
CONS_PARAGRAPH_TEXT = "div.norm.inline-element"
CONS_ANNEX_HEADING = 'p[class^="title-gr-seq-level-"]'


@dataclass(frozen=True)
class Selectors:
    """The CSS vocabulary one EUR-Lex dialect shares with the other."""

    article_heading: str
    annex_label: str
    annex_title: str


OJ = Selectors(
    article_heading="p.oj-ti-art",
    annex_label="p.oj-doc-ti",
    annex_title="p.oj-doc-ti",
)

CONS = Selectors(
    article_heading="p.title-article-norm",
    annex_label="p.title-annex-1",
    annex_title="p.title-gr-seq-level-1",
)


def clean(text: str) -> str:
    """Normalise whitespace and drop consolidated-text amendment marker glyphs."""
    return normalise(MARKERS.sub(" ", text))


def detect(tree: HTMLParser) -> Selectors:
    """OJ documents carry oj-* classes; consolidated ones carry norm."""
    if tree.css_first(".oj-normal") is not None:
        return OJ
    if tree.css_first("p.norm, div.norm") is not None:
        return CONS
    raise ParseError("unrecognised EUR-Lex dialect")


def prepare(html: str) -> HTMLParser:
    """Placeholder every base64 image and drop amendment banner blocks."""
    tree = HTMLParser(html)
    for image in tree.css("img"):
        if (image.attributes.get("src") or "").startswith("data:"):
            image.replace_with(FORMULA_PLACEHOLDER)
    for banner in tree.css("p.modref"):
        banner.decompose()
    return tree


def heading_number(node: Node, selector: str, pattern: re.Pattern[str]) -> str | None:
    heading = node.css_first(selector)
    if heading is None:
        return None
    match = pattern.search(clean(heading.text()))
    return match.group(1) if match else None


def article_section(node: Node, selectors: Selectors) -> Section:
    title = node.css_first(ARTICLE_TITLE)
    return Section(
        kind=SectionKind.ARTICLE,
        number=heading_number(node, selectors.article_heading, ARTICLE_NUMBER),
        title=clean(title.text()) if title else None,
    )


def parse_eurlex_html(html: str, ref: str, topic: str) -> ParsedDocument:
    """Parse one EUR-Lex document into the format-neutral section tree."""
    tree = prepare(html)
    selectors = detect(tree)
    sections = [article_section(node, selectors) for node in tree.css(ARTICLE)]
    if not sections:
        raise ParseError(f"{ref}: no articles found")
    return ParsedDocument(ref=ref, topic=topic, sections=tuple(sections))
