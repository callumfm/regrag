"""Cross-reference extraction over chunk text."""

import re
from collections.abc import Iterator

from app.core.models import FrozenModel

SECTOR = {"regulation": "R", "directive": "L", "decision": "D"}

ARTICLE_REF = re.compile(r"Article\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?")
ANNEX_REF = re.compile(r"Annex\s+([IVXLC]+|\d+)")
INSTRUMENT_REF = re.compile(
    r"(Regulation|Directive|Decision)\s*(?:\((?:EU|EC|EEC|Euratom)\)\s*)?(No\s*)?"
    r"(\d{1,4})/(\d{1,4})(?:/\w{2,7})?",
    re.IGNORECASE,
)
QUALIFIER = re.compile(r"^\s+(?:of|to|in)\s+(?:that\s+|the\s+)?$")

Span = tuple[int, int]


class Reference(FrozenModel):
    """One cross-reference; instrument is None when the target is this document."""

    raw: str
    instrument: str | None = None
    article: str | None = None
    paragraph: str | None = None
    annex: str | None = None


def celex(kind: str, numbered: bool, first: str, second: str) -> str:
    """CELEX id: 'No 765/2008' is number then year, '2015/757' is year then number."""
    number, year = (first, second) if numbered else (second, first)
    return f"3{year}{SECTOR[kind.lower()]}{number.zfill(4)}"


def instruments(text: str) -> list[tuple[Span, str]]:
    """Every instrument mention as its span and resolved CELEX id."""
    return [
        (m.span(), celex(m.group(1), bool(m.group(2)), m.group(3), m.group(4)))
        for m in INSTRUMENT_REF.finditer(text)
    ]


def divisions(text: str) -> Iterator[tuple[re.Match[str], dict[str, str | None]]]:
    """Article and annex mentions in order of appearance, with their Reference fields."""
    matches = [
        (m, {"article": m.group(1), "paragraph": m.group(2)}) for m in ARTICLE_REF.finditer(text)
    ]
    matches += [(m, {"annex": m.group(1)}) for m in ANNEX_REF.finditer(text)]
    yield from sorted(matches, key=lambda pair: pair[0].start())


def qualifies(text: str, division: re.Match[str], span: Span) -> bool:
    """Whether an instrument at span is the one a division belongs to ('Article 6 of X')."""
    return span[0] >= division.end() and QUALIFIER.match(text[division.end() : span[0]]) is not None


def extract_references(text: str) -> tuple[Reference, ...]:
    """Structured cross-references found in the text, deduplicated."""
    mentions = instruments(text)
    found: dict[Reference, None] = {}
    attributed: set[Span] = set()

    for match, fields in divisions(text):
        owner = next((pair for pair in mentions if qualifies(text, match, pair[0])), None)
        if owner is None:
            found[Reference(raw=match.group(0), **fields)] = None
            continue
        (start, end), instrument = owner
        attributed.add((start, end))
        found[Reference(raw=text[match.start() : end], instrument=instrument, **fields)] = None

    for (start, end), instrument in mentions:
        if (start, end) not in attributed:
            found[Reference(raw=text[start:end], instrument=instrument)] = None
    return tuple(found)
