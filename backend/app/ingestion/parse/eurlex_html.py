"""EUR-Lex HTML parser: one traversal over the OJ and consolidated dialects."""

from selectolax.parser import HTMLParser

from app.ingestion.exceptions import ParseError
from app.ingestion.parse.html.consolidated import CONSOLIDATED
from app.ingestion.parse.html.dialect import ANNEX_CONTAINER, ARTICLE_CONTAINER, Dialect
from app.ingestion.parse.html.oj import OJ
from app.ingestion.parse.html.sections import build_annex, build_article
from app.ingestion.parse.models import ParsedDocument
from app.ingestion.parse.text import FORMULA_PLACEHOLDER

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


def parse_eurlex_html(html: str, celex: str, topic: str) -> ParsedDocument:
    """Parse one EUR-Lex document into the format-neutral section tree."""
    tree = prepare(html)
    dialect = detect(tree)
    articles = [build_article(node, dialect) for node in tree.css(ARTICLE_CONTAINER)]
    if not articles:
        raise ParseError(f"{celex}: no articles found")
    annexes = [build_annex(node, dialect) for node in tree.css(ANNEX_CONTAINER)]
    return ParsedDocument(celex=celex, topic=topic, sections=tuple(articles + annexes))
