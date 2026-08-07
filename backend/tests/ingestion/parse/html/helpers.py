"""The two dialect fixtures, and tree helpers shared by the EUR-Lex HTML parser tests."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from selectolax.parser import Node

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.parser import prepare
from app.ingestion.parse.models import ParsedDocument, Section

FIXTURES = Path(__file__).parent.parent / "fixtures"
FUELEU_HTML = (FIXTURES / "32023R1805.html").read_text()
MRV_HTML = (FIXTURES / "32015R0757.html").read_text()


def of_kind(sections: Iterable[Section], kind: SectionKind) -> list[Section]:
    return [section for section in sections if section.kind is kind]


def articles(document: ParsedDocument) -> list[Section]:
    return of_kind(document.sections, SectionKind.ARTICLE)


def annexes(document: ParsedDocument) -> list[Section]:
    return of_kind(document.sections, SectionKind.ANNEX)


def all_sections(sections: Iterable[Section]) -> Iterator[Section]:
    for section in sections:
        yield section
        yield from all_sections(section.children)


def subdivision(html: str, node_id: str) -> Node:
    node = prepare(html).css_first(f'div[id="{node_id}"]')
    assert node is not None
    return node
