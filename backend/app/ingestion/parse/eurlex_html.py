"""EUR-Lex HTML parser: one traversal over the OJ and consolidated dialects."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from selectolax.parser import HTMLParser, Node

from app.ingestion.enums import SectionKind
from app.ingestion.exceptions import ParseError
from app.ingestion.parse.models import ParsedDocument, Section
from app.ingestion.parse.text import (
    ANNEX_NUMBER_RE,
    ARTICLE_NUMBER_RE,
    FORMULA_PLACEHOLDER,
    LEADING_NUMBER_RE,
    clean_text,
)

PARAGRAPH_ID = re.compile(r"^\d+\.\d+$")
HEADING_LEVEL = re.compile(r"title-gr-seq-level-(\d+)")
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
OJ_DATA_TABLE = "table.oj-table"
CONS_DATA_TABLE = "table.borderOj"
GRID_CONTAINER = "div.grid-container"
GRID_MARKER = "div.grid-list-column-1"
GRID_BODY = "div.grid-list-column-2"


@dataclass(frozen=True)
class Dialect:
    """One EUR-Lex markup dialect: how to recognise it and where it keeps each part."""

    signature: str
    article_heading: str
    annex_label: str
    data_table: str
    annex_title: Callable[[Node], str | None]
    paragraphs: Callable[[Node], list[Node]]
    paragraph: Callable[[Node], Section]
    annex_headings: str | None = None


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
        cells = tuple(clean_text(cell.text()) for cell in row.css(CELL))
        if cells:
            rows.append(cells)
    return tuple(rows)


def extract_tables(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """Take data tables out of the tree, so their cells never re-appear as prose."""
    tables = node.css(dialect.data_table)
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
    return " ".join(text for text in (clean_text(n.text()) for n in nodes if n is not None) if text)


def grid_line(node: Node) -> str:
    """A consolidated list row: the '(a)' marker column joined to its text column."""
    return join_texts((node.css_first(GRID_MARKER), node.css_first(GRID_BODY)))


def row_line(node: Node) -> str:
    """A layout table row: its cells joined, so a bullet stays with the text it marks."""
    return join_texts(node.css(CELL))


@dataclass(frozen=True)
class Heading:
    """A sub-heading in a line stream, at the title-gr-seq depth it was written at."""

    level: int
    title: str


type Line = Heading | str


def heading_line(node: Node, selector: str | None) -> Heading | None:
    """The sub-heading this block introduces, or None if the block is prose."""
    if selector is None or not node.css_matches(selector):
        return None
    match = HEADING_LEVEL.search(node.attributes.get("class") or "")
    return Heading(level=int(match.group(1)) if match else 1, title=clean_text(node.text()))


def collect_lines(node: Node, lines: list[Line], headings: str | None = None) -> None:
    """Walk in document order, emitting one line per block and per flattened list row."""
    for child in node.iter(include_text=False):
        heading = heading_line(child, headings)
        if heading is not None:
            lines.append(heading)
            continue
        if child.css_matches(GRID_CONTAINER):
            line = grid_line(child)
        elif child.tag == "tr":
            line = row_line(child)
        elif child.tag in ("p", "td"):
            line = clean_text(child.text())
        else:
            collect_lines(child, lines, headings)
            continue
        if line and line not in lines[-1:]:
            lines.append(line)


def block_text(node: Node) -> str:
    """Join a node's text block by block, so flattened list rows stay on separate lines."""
    lines: list[Line] = []
    collect_lines(node, lines)
    text = "\n".join(line for line in lines if isinstance(line, str))
    return text if text else clean_text(node.text())


def prose_lines(node: Node, *headings: str) -> list[str]:
    """The node's text lines, minus any line that repeats one of its own heading lines."""
    skip = {clean_text(n.text()) for selector in headings for n in node.css(selector)}
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
    match = LEADING_NUMBER_RE.match(text)
    if match:
        number, text = match.group(1), text[match.end() :]
    return Section(kind=SectionKind.PARAGRAPH, number=number, text=text)


def cons_paragraph(node: Node) -> Section:
    marker = node.css_first(CONS_PARAGRAPH_NUMBER)
    body = node.css_first(CONS_PARAGRAPH_TEXT)
    number = LEADING_NUMBER_RE.match(clean_text(marker.text()) if marker else "")
    return Section(
        kind=SectionKind.PARAGRAPH,
        number=number.group(1) if number else None,
        text=block_text(body) if body is not None else "",
    )


def oj_annex_title(node: Node) -> str | None:
    """OJ annexes repeat the same class for the label and the title that follows it."""
    labels = node.css(OJ_ANNEX_LABEL)
    return clean_text(labels[1].text()) if len(labels) > 1 else None


def cons_annex_title(node: Node) -> str | None:
    title = node.css_first(CONS_ANNEX_TITLE)
    return clean_text(title.text()) if title is not None else None


OJ = Dialect(
    signature=".oj-normal",
    article_heading="p.oj-ti-art",
    annex_label=OJ_ANNEX_LABEL,
    data_table=OJ_DATA_TABLE,
    annex_title=oj_annex_title,
    paragraphs=oj_paragraphs,
    paragraph=oj_paragraph,
)

CONS = Dialect(
    signature="p.norm, div.norm",
    article_heading="p.title-article-norm",
    annex_label="p.title-annex-1",
    data_table=CONS_DATA_TABLE,
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
    match = pattern.search(clean_text(heading.text()))
    return match.group(1) if match else None


def article_section(node: Node, dialect: Dialect) -> Section:
    title = node.css_first(ARTICLE_TITLE)
    return Section(
        kind=SectionKind.ARTICLE,
        number=heading_number(node, dialect.article_heading, ARTICLE_NUMBER_RE),
        title=clean_text(title.text()) if title else None,
        children=paragraph_sections(node, dialect),
    )


@dataclass
class Frame:
    """One open heading while folding a line stream: its own prose, then its subsections."""

    level: int
    title: str | None
    lines: list[str]
    children: list[Section]

    def flush(self) -> None:
        """Bank the prose read so far as a paragraph, ahead of any subsection after it."""
        if self.lines:
            self.children.append(Section(kind=SectionKind.PARAGRAPH, text="\n".join(self.lines)))
            self.lines.clear()

    def section(self) -> Section:
        self.flush()
        return Section(kind=SectionKind.HEADING, title=self.title, children=tuple(self.children))


def nest(lines: Iterable[Line]) -> tuple[Section, ...]:
    """Fold a line stream into sections, each heading owning the prose written beneath it."""
    stack = [Frame(level=1, title=None, lines=[], children=[])]

    def close(level: int) -> None:
        while stack[-1].level >= level:
            done = stack.pop()
            stack[-1].children.append(done.section())

    for line in lines:
        if isinstance(line, str):
            stack[-1].lines.append(line)
        elif line.level > 1:
            close(line.level)
            stack[-1].flush()
            stack.append(Frame(level=line.level, title=line.title, lines=[], children=[]))
    close(2)
    stack[0].flush()
    return tuple(stack[0].children)


def annex_body(node: Node, dialect: Dialect) -> tuple[Section, ...]:
    """The annex prose nested under its own sub-headings, with its label lines removed."""
    lines: list[Line] = []
    collect_lines(node, lines, dialect.annex_headings)
    skip = {clean_text(label.text()) for label in node.css(dialect.annex_label)}
    return nest(line for line in lines if isinstance(line, Heading) or line not in skip)


def annex_section(node: Node, dialect: Dialect) -> Section:
    """OJ annexes are flat; consolidated ones nest by title-gr-seq level."""
    tables = extract_tables(node, dialect)
    return Section(
        kind=SectionKind.ANNEX,
        number=heading_number(node, dialect.annex_label, ANNEX_NUMBER_RE),
        title=dialect.annex_title(node),
        children=tables + annex_body(node, dialect),
    )


def parse_eurlex_html(html: str, celex: str, topic: str) -> ParsedDocument:
    """Parse one EUR-Lex document into the format-neutral section tree."""
    tree = prepare(html)
    dialect = detect(tree)
    articles = [article_section(node, dialect) for node in tree.css(ARTICLE)]
    if not articles:
        raise ParseError(f"{celex}: no articles found")
    annexes = [annex_section(node, dialect) for node in tree.css(ANNEX)]
    return ParsedDocument(celex=celex, topic=topic, sections=tuple(articles + annexes))
