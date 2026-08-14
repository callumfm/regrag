"""Tree helpers shared by the EUR-Lex HTML parser tests."""

from collections.abc import Iterable, Iterator

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.document import prepare
from app.ingestion.parse.models import Section


def of_kind(sections: Iterable[Section], kind: SectionKind) -> list[Section]:
    return [section for section in sections if section.kind is kind]


def articles(sections: Iterable[Section]) -> list[Section]:
    return of_kind(sections, SectionKind.ARTICLE)


def annexes(sections: Iterable[Section]) -> list[Section]:
    return of_kind(sections, SectionKind.ANNEX)


def all_sections(sections: Iterable[Section]) -> Iterator[Section]:
    for section in sections:
        yield section
        yield from all_sections(section.children)


def subdivision(html: str, node_id: str) -> Node:
    """One container of the document, with the markup stripped as the parser would."""
    node = prepare(html).css_first(f'div[id="{node_id}"]')
    assert node is not None
    return node
