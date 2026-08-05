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
HEADING_LEVEL = re.compile(r"title-gr-seq-level-(\d+)")
MARKERS = re.compile(r"[▼►]\s*[A-Z]+\d*|◄")
EMPTY_PARENS = re.compile(r"\s*\(\s*\)")
FOOTNOTE_REF = "span.superscript, span.oj-super"

ARTICLE = "div.eli-subdivision[id^=art_]"
ANNEX = "div[id^=anx_]"
ARTICLE_TITLE = "div.eli-title p"

OJ_PARAGRAPH = "div[id]"
CONS_PARAGRAPH = "div.norm"
CONS_PARAGRAPH_NUMBER = "span.no-parag"
CONS_PARAGRAPH_TEXT = "div.norm.inline-element"
CONS_ANNEX_HEADING = 'p[class^="title-gr-seq-level-"]'
DATA_TABLE = "table.oj-table"
GRID_CONTAINER = "grid-container"
GRID_MARKER = "div.grid-list-column-1"
GRID_BODY = "div.grid-list-column-2"


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
    """Normalise whitespace, dropping amendment glyphs and emptied footnote brackets."""
    return normalise(EMPTY_PARENS.sub("", MARKERS.sub(" ", text)))


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
    for marker in tree.css(FOOTNOTE_REF):
        marker.decompose()
    return tree


def table_rows(node: Node) -> tuple[tuple[str, ...], ...]:
    rows = []
    for row in node.css("tr"):
        cells = tuple(clean(cell.text()) for cell in row.css("td, th"))
        if cells:
            rows.append(cells)
    return tuple(rows)


def extract_tables(node: Node) -> tuple[Section, ...]:
    """Take data tables out of the tree, so their cells never re-appear as prose."""
    tables = node.css(DATA_TABLE)
    sections = tuple(
        Section(kind=SectionKind.TABLE, rows=rows)
        for rows in (table_rows(table) for table in tables)
        if rows
    )
    for table in tables:
        table.decompose()
    return sections


def grid_line(node: Node) -> str:
    """A consolidated list row: the '(a)' marker column joined to its text column."""
    columns = (node.css_first(GRID_MARKER), node.css_first(GRID_BODY))
    return " ".join(clean(c.text()) for c in columns if c is not None and clean(c.text()))


def row_line(node: Node) -> str:
    """A layout table row: its cells joined, so a bullet stays with the text it marks."""
    return " ".join(text for text in (clean(c.text()) for c in node.css("td, th")) if text)


def collect_lines(node: Node, lines: list[str]) -> None:
    """Walk in document order, emitting one line per block and per flattened list row."""
    for child in node.iter(include_text=False):
        if GRID_CONTAINER in (child.attributes.get("class") or ""):
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


def article_body_text(node: Node, selectors: Selectors) -> str:
    """Article text with its heading and title lines removed, for unnumbered articles."""
    skip = {clean(n.text()) for n in node.css(selectors.article_heading)}
    skip |= {clean(n.text()) for n in node.css(ARTICLE_TITLE)}
    lines = [line for line in block_text(node).split("\n") if line and line not in skip]
    return "\n".join(lines)


def paragraph_sections(node: Node, selectors: Selectors) -> tuple[Section, ...]:
    """Numbered paragraphs, falling back to one unnumbered paragraph for the whole article."""
    if selectors is OJ:
        sections = [oj_paragraph(child) for child in oj_paragraphs(node)]
    else:
        sections = [cons_paragraph(child) for child in cons_paragraphs(node)]
    if sections:
        return tuple(sections)
    body = article_body_text(node, selectors)
    return (Section(kind=SectionKind.PARAGRAPH, text=body),) if body else ()


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
        children=paragraph_sections(node, selectors),
    )


def cons_heading_tree(node: Node) -> tuple[Section, ...]:
    """Nest title-gr-seq-level-N headings, where level 1 is the annex title itself."""
    stack: list[tuple[int, list[Section], str | None]] = [(1, [], None)]
    for heading in node.css(CONS_ANNEX_HEADING):
        match = HEADING_LEVEL.search(heading.attributes.get("class") or "")
        level = int(match.group(1)) if match else 1
        if level < 2:
            continue
        while stack[-1][0] >= level:
            _, children, title = stack.pop()
            stack[-1][1].append(
                Section(kind=SectionKind.HEADING, title=title, children=tuple(children))
            )
        stack.append((level, [], clean(heading.text())))
    while len(stack) > 1:
        _, children, title = stack.pop()
        stack[-1][1].append(
            Section(kind=SectionKind.HEADING, title=title, children=tuple(children))
        )
    return tuple(stack[0][1])


def annex_section(node: Node, selectors: Selectors) -> Section:
    """OJ annexes are flat; consolidated ones nest by title-gr-seq level."""
    labels = node.css(selectors.annex_label)
    number = None
    if labels:
        match = ANNEX_NUMBER.search(clean(labels[0].text()))
        number = match.group(1) if match else None
    if selectors is OJ:
        title = clean(labels[1].text()) if len(labels) > 1 else None
        children = extract_tables(node)
    else:
        title_node = node.css_first(selectors.annex_title)
        title = clean(title_node.text()) if title_node else None
        children = extract_tables(node) + cons_heading_tree(node)
    return Section(kind=SectionKind.ANNEX, number=number, title=title, children=children)


def parse_eurlex_html(html: str, ref: str, topic: str) -> ParsedDocument:
    """Parse one EUR-Lex document into the format-neutral section tree."""
    tree = prepare(html)
    selectors = detect(tree)
    articles = [article_section(node, selectors) for node in tree.css(ARTICLE)]
    if not articles:
        raise ParseError(f"{ref}: no articles found")
    annexes = [annex_section(node, selectors) for node in tree.css(ANNEX)]
    return ParsedDocument(ref=ref, topic=topic, sections=tuple(articles + annexes))
