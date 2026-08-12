"""One line per block: EUR-Lex renders lists as tables, so blocks are flattened by hand."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import Section
from app.ingestion.parse.text import clean_text

CELL = "td, th"
GRID_CONTAINER = "div.grid-container"
GRID_MARKER = "div.grid-list-column-1"
GRID_BODY = "div.grid-list-column-2"
TITLE_LEVEL = 1


@dataclass(frozen=True)
class Subheading:
    """A sub-heading in a line stream, at the depth its own class was written at."""

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


def heading_texts(node: Node, *selectors: str) -> set[str]:
    """The cleaned text of the node's own heading lines, for filtering back out of its prose."""
    return {clean_text(n.text()) for selector in selectors for n in node.css(selector)}


def read_subheading(node: Node, level_pattern: re.Pattern[str] | None) -> Subheading | None:
    """The sub-heading this paragraph introduces; a container carrying the class opens one."""
    if node.tag != "p" or level_pattern is None:
        return None
    match = level_pattern.search(node.attributes.get("class") or "")
    return Subheading(level=int(match.group(1)), title=clean_text(node.text())) if match else None


def collect_lines(node: Node, level_pattern: re.Pattern[str] | None = None) -> list[Line]:
    """Walk in document order, one line per block, falling back to the node's own text."""
    lines: list[Line] = []
    _collect(node, lines, level_pattern)
    if not lines and (text := clean_text(node.text())):
        lines.append(text)
    return lines


def _collect(node: Node, lines: list[Line], level_pattern: re.Pattern[str] | None) -> None:
    """Recurse into containers, emitting one line per text block."""
    for child in node.iter(include_text=False):
        subheading = read_subheading(child, level_pattern)
        if subheading is not None:
            if subheading.level > TITLE_LEVEL:
                lines.append(subheading)
            continue
        if child.css_matches(GRID_CONTAINER):
            line = grid_line(child)
        elif child.tag == "tr":
            line = row_line(child)
        elif child.tag in ("p", "td"):
            line = clean_text(child.text())
        else:
            _collect(child, lines, level_pattern)
            continue
        if line:
            lines.append(line)


def block_text(node: Node) -> str:
    """Join a node's text block by block, so flattened list rows stay on separate lines."""
    return "\n".join(line for line in collect_lines(node) if isinstance(line, str))


def prose_lines(node: Node, *headings: str) -> list[str]:
    """The node's text lines, minus any line that repeats one of its own heading lines."""
    skip = heading_texts(node, *headings)
    return [line for line in block_text(node).split("\n") if line and line not in skip]


@dataclass
class OpenSubheading:
    """A sub-heading still being filled: its own prose, then its subsections."""

    level: int
    title: str | None
    lines: list[str] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)

    def flush_prose(self) -> None:
        """Bank the prose read so far as a paragraph, ahead of any subsection after it."""
        if self.lines:
            self.children.append(Section(kind=SectionKind.PARAGRAPH, text="\n".join(self.lines)))
            self.lines.clear()

    def close(self) -> Section:
        """Bank any prose still open and return the finished heading section."""
        self.flush_prose()
        return Section(kind=SectionKind.HEADING, title=self.title, children=tuple(self.children))


def nest_under_subheadings(lines: Iterable[Line]) -> tuple[Section, ...]:
    """Fold a line stream into sections, each sub-heading owning the prose beneath it."""
    stack = [OpenSubheading(level=0, title=None)]

    def unwind(level: int) -> None:
        while stack[-1].level >= level:
            done = stack.pop()
            stack[-1].children.append(done.close())

    for line in lines:
        if isinstance(line, str):
            stack[-1].lines.append(line)
        else:
            unwind(line.level)
            stack[-1].flush_prose()
            stack.append(OpenSubheading(level=line.level, title=line.title))
    unwind(1)
    stack[0].flush_prose()
    return tuple(stack[0].children)
