"""EUR-Lex HTML parser: one traversal over the OJ and consolidated dialects."""

import re
from collections.abc import Callable, Iterable
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
HEADING_LEVEL = re.compile(r"title-gr-seq-level-(\d+)")
MARKERS = re.compile(r"[▼►]\s*[A-Z]+\d*|◄")
EMPTY_PARENS = re.compile(r"\s*\(\s*\)")
MARKER_GLYPHS = "▼►◄"
FOOTNOTE_REF = "span.superscript, span.oj-super"
FOOTNOTE = "p.footnote, p.oj-note, div[id^=fnp]"
DROP = ("p.modref", FOOTNOTE_REF, FOOTNOTE)

ARTICLE = "div.eli-subdivision[id^=art_]"
ANNEX = "div[id^=anx_]"
ARTICLE_TITLE = "div.eli-title p"
CELL = "td, th"

OJ_PARAGRAPH = "div[id]"
OJ_ANNEX_LABEL = "p.oj-doc-ti"
CONS_PARAGRAPH = "div.norm"
CONS_PARAGRAPH_NUMBER = "span.no-parag"
CONS_PARAGRAPH_TEXT = "div.norm.inline-element"
CONS_ANNEX_TITLE = "p.title-gr-seq-level-1"
CONS_ANNEX_HEADING = 'p[class^="title-gr-seq-level-"]'
DATA_TABLE = "table.oj-table"
GRID_CONTAINER = "div.grid-container"
GRID_MARKER = "div.grid-list-column-1"
GRID_BODY = "div.grid-list-column-2"


def clean(text: str) -> str:
    """Normalise whitespace, dropping amendment glyphs and emptied footnote brackets."""
    if any(glyph in text for glyph in MARKER_GLYPHS):
        text = MARKERS.sub(" ", text)
    if "(" in text:
        text = EMPTY_PARENS.sub("", text)
    return normalise(text)


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


def table_rows(node: Node) -> tuple[tuple[str, ...], ...]:
    rows = []
    for row in node.css("tr"):
        cells = tuple(clean(cell.text()) for cell in row.css(CELL))
        if cells:
            rows.append(cells)
    return tuple(rows)


def extract_tables(node: Node) -> tuple[Section, ...]:
    """Take data tables out of the tree, so their cells never re-appear as prose."""
    tables = node.css(DATA_TABLE)
    sections = tuple(
        Section(kind=SectionKind.TABLE, rows=rows)
        for table in tables
        if (rows := table_rows(table))
    )
    for table in tables:
        table.decompose()
    return sections


def join_texts(nodes: Iterable[Node | None]) -> str:
    """Clean each node's text and join the non-empty ones with a space."""
    return " ".join(text for text in (clean(n.text()) for n in nodes if n is not None) if text)


def grid_line(node: Node) -> str:
    """A consolidated list row: the '(a)' marker column joined to its text column."""
    return join_texts((node.css_first(GRID_MARKER), node.css_first(GRID_BODY)))


def row_line(node: Node) -> str:
    """A layout table row: its cells joined, so a bullet stays with the text it marks."""
    return join_texts(node.css(CELL))


def collect_lines(node: Node, lines: list[str]) -> None:
    """Walk in document order, emitting one line per block and per flattened list row."""
    for child in node.iter(include_text=False):
        if child.css_matches(GRID_CONTAINER):
            line = grid_line(child)
        elif child.tag == "tr":
            line = row_line(child)
        elif child.tag in ("p", "td"):
            line = clean(child.text())
        else:
            collect_lines(child, lines)
            continue
        if line and line not in lines[-1:]:
            lines.append(line)


def block_text(node: Node) -> str:
    """Join a node's text block by block, so flattened list rows stay on separate lines."""
    lines: list[str] = []
    collect_lines(node, lines)
    return "\n".join(lines) if lines else clean(node.text())


def prose_lines(node: Node, *headings: str) -> list[str]:
    """The node's text lines, minus any line that repeats one of its own heading lines."""
    skip = {clean(n.text()) for selector in headings for n in node.css(selector)}
    return [line for line in block_text(node).split("\n") if line and line not in skip]


def oj_paragraphs(node: Node) -> list[Node]:
    """OJ paragraph containers have ids like 004.001; Node.css matches self, so exclude it."""
    own_id = node.attributes.get("id")
    return [
        child
        for child in node.css(OJ_PARAGRAPH)
        if child.attributes.get("id") != own_id
        and PARAGRAPH_ID.match(child.attributes.get("id") or "")
    ]


def cons_paragraphs(node: Node) -> list[Node]:
    return [child for child in node.css(CONS_PARAGRAPH) if child.css_first(CONS_PARAGRAPH_NUMBER)]


def oj_paragraph(node: Node) -> Section:
    number, text = None, block_text(node)
    match = LEADING_NUMBER.match(text)
    if match:
        number, text = match.group(1), text[match.end() :]
    return Section(kind=SectionKind.PARAGRAPH, number=number, text=text)


def cons_paragraph(node: Node) -> Section:
    marker = node.css_first(CONS_PARAGRAPH_NUMBER)
    body = node.css_first(CONS_PARAGRAPH_TEXT)
    number = LEADING_NUMBER.match(clean(marker.text()) if marker else "")
    return Section(
        kind=SectionKind.PARAGRAPH,
        number=number.group(1) if number else None,
        text=block_text(body) if body is not None else "",
    )


def oj_annex_title(node: Node) -> str | None:
    """OJ annexes repeat the same class for the label and the title that follows it."""
    labels = node.css(OJ_ANNEX_LABEL)
    return clean(labels[1].text()) if len(labels) > 1 else None


def cons_annex_title(node: Node) -> str | None:
    title = node.css_first(CONS_ANNEX_TITLE)
    return clean(title.text()) if title is not None else None


@dataclass(frozen=True)
class Dialect:
    """One EUR-Lex markup dialect: how to recognise it and where it keeps each part."""

    signature: str
    article_heading: str
    annex_label: str
    annex_title: Callable[[Node], str | None]
    paragraphs: Callable[[Node], list[Node]]
    paragraph: Callable[[Node], Section]
    annex_headings: str | None = None


OJ = Dialect(
    signature=".oj-normal",
    article_heading="p.oj-ti-art",
    annex_label=OJ_ANNEX_LABEL,
    annex_title=oj_annex_title,
    paragraphs=oj_paragraphs,
    paragraph=oj_paragraph,
)

CONS = Dialect(
    signature="p.norm, div.norm",
    article_heading="p.title-article-norm",
    annex_label="p.title-annex-1",
    annex_title=cons_annex_title,
    paragraphs=cons_paragraphs,
    paragraph=cons_paragraph,
    annex_headings=CONS_ANNEX_HEADING,
)

DIALECTS = (OJ, CONS)


def detect(tree: HTMLParser) -> Dialect:
    """OJ documents carry oj-* classes; consolidated ones carry norm."""
    for dialect in DIALECTS:
        if tree.css_first(dialect.signature) is not None:
            return dialect
    raise ParseError("unrecognised EUR-Lex dialect")


def paragraph_sections(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """Numbered paragraphs, falling back to one unnumbered paragraph for the whole article."""
    sections = [dialect.paragraph(child) for child in dialect.paragraphs(node)]
    if sections:
        return tuple(sections)
    lines = prose_lines(node, dialect.article_heading, ARTICLE_TITLE)
    return (Section(kind=SectionKind.PARAGRAPH, text="\n".join(lines)),) if lines else ()


def heading_number(node: Node, selector: str, pattern: re.Pattern[str]) -> str | None:
    heading = node.css_first(selector)
    if heading is None:
        return None
    match = pattern.search(clean(heading.text()))
    return match.group(1) if match else None


def article_section(node: Node, dialect: Dialect) -> Section:
    title = node.css_first(ARTICLE_TITLE)
    return Section(
        kind=SectionKind.ARTICLE,
        number=heading_number(node, dialect.article_heading, ARTICLE_NUMBER),
        title=clean(title.text()) if title else None,
        children=paragraph_sections(node, dialect),
    )


def heading_tree(node: Node, selector: str) -> tuple[Section, ...]:
    """Nest title-gr-seq-level-N headings, where level 1 is the annex title itself."""
    stack: list[tuple[int, list[Section], str | None]] = [(1, [], None)]

    def close(level: int) -> None:
        while stack[-1][0] >= level:
            _, children, title = stack.pop()
            stack[-1][1].append(
                Section(kind=SectionKind.HEADING, title=title, children=tuple(children))
            )

    for heading in node.css(selector):
        match = HEADING_LEVEL.search(heading.attributes.get("class") or "")
        level = int(match.group(1)) if match else 1
        if level < 2:
            continue
        close(level)
        stack.append((level, [], clean(heading.text())))
    close(2)
    return tuple(stack[0][1])


def annex_body(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """The annex prose, with its own label, title and sub-heading lines removed."""
    headings = (dialect.annex_headings,) if dialect.annex_headings else ()
    lines = prose_lines(node, dialect.annex_label, *headings)
    return (Section(kind=SectionKind.PARAGRAPH, text="\n".join(lines)),) if lines else ()


def annex_section(node: Node, dialect: Dialect) -> Section:
    """OJ annexes are flat; consolidated ones nest by title-gr-seq level."""
    tables = extract_tables(node)
    headings = heading_tree(node, dialect.annex_headings) if dialect.annex_headings else ()
    return Section(
        kind=SectionKind.ANNEX,
        number=heading_number(node, dialect.annex_label, ANNEX_NUMBER),
        title=dialect.annex_title(node),
        children=tables + annex_body(node, dialect) + headings,
    )


def parse_eurlex_html(html: str, ref: str, topic: str) -> ParsedDocument:
    """Parse one EUR-Lex document into the format-neutral section tree."""
    tree = prepare(html)
    dialect = detect(tree)
    articles = [article_section(node, dialect) for node in tree.css(ARTICLE)]
    if not articles:
        raise ParseError(f"{ref}: no articles found")
    annexes = [annex_section(node, dialect) for node in tree.css(ANNEX)]
    return ParsedDocument(ref=ref, topic=topic, sections=tuple(articles + annexes))
