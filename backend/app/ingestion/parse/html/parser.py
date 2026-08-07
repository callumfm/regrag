"""EUR-Lex HTML to a section tree, in one pass over either markup dialect.

Order is load-bearing. drop_non_legal_markup runs before any text is cleaned,
because removing a footnote superscript leaves an empty "()" behind and
clean_text is what strips it. detach_data_tables likewise runs before annex
prose is read, so table cells never re-appear as prose.
"""

from selectolax.parser import HTMLParser

from app.ingestion.exceptions import ParseError
from app.ingestion.parse.html.consolidated import CONSOLIDATED
from app.ingestion.parse.html.dialect import Dialect
from app.ingestion.parse.html.oj import OJ
from app.ingestion.parse.html.sections import build_annex, build_article
from app.ingestion.parse.models import ParsedDocument
from app.ingestion.parse.text import FORMULA_PLACEHOLDER

ARTICLE_CONTAINER = "div.eli-subdivision[id^=art_]"
ANNEX_CONTAINER = "div[id^=anx_]"

AMENDMENT_REF = "p.modref"
FOOTNOTE_MARKER = "span.superscript, span.oj-super"
FOOTNOTE_BLOCK = "p.footnote, p.oj-note, div[id^=fnp]"
NON_LEGAL_MARKUP = (AMENDMENT_REF, FOOTNOTE_MARKER, FOOTNOTE_BLOCK)

DIALECTS = (OJ, CONSOLIDATED)


def placeholder_formula_images(tree: HTMLParser) -> None:
    """Stand every base64 formula image down to a marker the chunker can carry."""
    for image in tree.css("img"):
        if (image.attributes.get("src") or "").startswith("data:"):
            image.replace_with(FORMULA_PLACEHOLDER)


def drop_non_legal_markup(tree: HTMLParser) -> None:
    """Remove amendment references, footnote markers and footnote blocks."""
    for selector in NON_LEGAL_MARKUP:
        for node in tree.css(selector):
            node.decompose()


def detect_dialect(tree: HTMLParser) -> Dialect:
    """OJ documents carry oj-* classes; consolidated ones carry norm."""
    for dialect in DIALECTS:
        if tree.css_first(dialect.signature) is not None:
            return dialect
    raise ParseError("unrecognised EUR-Lex dialect")


def parse_eurlex_html(html: str, celex: str, topic: str) -> ParsedDocument:
    """Parse one EUR-Lex document into the format-neutral section tree."""
    tree = HTMLParser(html)
    placeholder_formula_images(tree)
    drop_non_legal_markup(tree)
    dialect = detect_dialect(tree)

    articles = [build_article(node, dialect) for node in tree.css(ARTICLE_CONTAINER)]
    if not articles:
        raise ParseError(f"{celex}: no articles found")
    annexes = [build_annex(node, dialect) for node in tree.css(ANNEX_CONTAINER)]
    return ParsedDocument(celex=celex, topic=topic, sections=tuple(articles + annexes))
