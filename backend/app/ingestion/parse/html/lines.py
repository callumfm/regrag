"""One line per block: EUR-Lex renders lists as tables, so blocks are flattened by hand."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.dialect import CELL
from app.ingestion.parse.models import Section
from app.ingestion.parse.text import clean_text

GRID_CONTAINER = "div.grid-container"
GRID_MARKER = "div.grid-list-column-1"
GRID_BODY = "div.grid-list-column-2"
HEADING_LEVEL_RE = re.compile(r"title-gr-seq-level-(\d+)")


@dataclass(frozen=True)
class Subheading:
    """A sub-heading in a line stream, at the title-gr-seq depth it was written at."""

    level: int
    title: str


type Line = Subheading | str


def join_texts(nodes: Iterable[Node | None]) -> str:
    """Clean each node's text and join the non-empty ones with a space."""
    return " ".join(text for text in (clean_text(n.text()) for n in nodes if n is not None) if text)


def grid_line(node: Node) -> str:
    """A consolidated list row: the '(a)' marker column joined to its text column."""
    return join_texts((node.css_first(GRID_MARKER), node.css_first(GRID_BODY)))


def row_line(node: Node) -> str:
    """A layout table row: its cells joined, so a bullet stays with the text it marks."""
    return join_texts(node.css(CELL))


def read_subheading(node: Node, selector: str | None) -> Subheading | None:
    """The sub-heading this block introduces, or None if the block is prose."""
    if selector is None or not node.css_matches(selector):
        return None
    match = HEADING_LEVEL_RE.search(node.attributes.get("class") or "")
    return Subheading(level=int(match.group(1)) if match else 1, title=clean_text(node.text()))


def collect_lines(node: Node, lines: list[Line], subheadings: str | None = None) -> None:
    """Walk in document order, emitting one line per block and per flattened list row."""
    for child in node.iter(include_text=False):
        subheading = read_subheading(child, subheadings)
        if subheading is not None:
            lines.append(subheading)
            continue
        if child.css_matches(GRID_CONTAINER):
            line = grid_line(child)
        elif child.tag == "tr":
            line = row_line(child)
        elif child.tag in ("p", "td"):
            line = clean_text(child.text())
        else:
            collect_lines(child, lines, subheadings)
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


@dataclass
class OpenSubheading:
    """A sub-heading still being filled: its own prose, then its subsections."""

    level: int
    title: str | None
    lines: list[str]
    children: list[Section]

    def flush_prose(self) -> None:
        """Bank the prose read so far as a paragraph, ahead of any subsection after it."""
        if self.lines:
            self.children.append(Section(kind=SectionKind.PARAGRAPH, text="\n".join(self.lines)))
            self.lines.clear()

    def close(self) -> Section:
        self.flush_prose()
        return Section(kind=SectionKind.HEADING, title=self.title, children=tuple(self.children))


def nest_under_subheadings(lines: Iterable[Line]) -> tuple[Section, ...]:
    """Fold a line stream into sections, each sub-heading owning the prose beneath it."""
    stack = [OpenSubheading(level=1, title=None, lines=[], children=[])]

    def unwind(level: int) -> None:
        while stack[-1].level >= level:
            done = stack.pop()
            stack[-1].children.append(done.close())

    for line in lines:
        if isinstance(line, str):
            stack[-1].lines.append(line)
        elif line.level > 1:
            unwind(line.level)
            stack[-1].flush_prose()
            stack.append(OpenSubheading(level=line.level, title=line.title, lines=[], children=[]))
    unwind(2)
    stack[0].flush_prose()
    return tuple(stack[0].children)
