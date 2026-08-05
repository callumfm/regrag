"""Cross-reference extraction over chunk text."""

import re

from app.core.models import FrozenModel
from app.ingestion import celex

ARTICLE_REF = re.compile(r"Articles?\s+(\d+[a-z]?)(?:\((\d+[a-z]?)\))?")
ARTICLE_TAIL = re.compile(r"\s*(?:,|and)\s+(\d{1,3}[a-z]?)\b(?:\((\d+[a-z]?)\))?")
ANNEX_REF = re.compile(r"Annexe?s?\s+([IVXLC]+|\d+)")
ANNEX_TAIL = re.compile(r"\s*(?:,|and)\s+([IVXLC]{1,4}|\d{1,2})\b")
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


class Division(FrozenModel):
    """One member of an article or annex mention, ending where its enumeration does."""

    start: int
    end: int
    reference: Reference


def order(numbered: bool, first: str, second: str) -> tuple[str, str]:
    """A '765/2008' pair as (number, year); EU drafting uses both orderings."""
    left, right = celex.as_year(first), celex.as_year(second)
    if left and right:
        return (second, first) if left >= right else (first, second)
    if right:
        return first, second
    if left:
        return second, first
    return (first, second) if numbered else (second, first)


def instruments(text: str) -> list[tuple[Span, str | None]]:
    """Every instrument mention as its span and CELEX id, None where it cannot resolve."""
    found: list[tuple[Span, str | None]] = []
    for match in INSTRUMENT_REF.finditer(text):
        number, year = order(bool(match.group(2)), match.group(3), match.group(4))
        try:
            found.append((match.span(), celex.build(match.group(1), year, number)))
        except ValueError:
            found.append((match.span(), None))
    return found


def enumerated(head: re.Match[str], text: str, tail: re.Pattern[str]) -> list[re.Match[str]]:
    """An 'Articles 6, 7 and 8' run as one match per member."""
    members = [head]
    while member := tail.match(text, members[-1].end()):
        members.append(member)
    return members


def cited(kind: str, number: str, paragraph: str | None = None) -> str:
    """One member of an enumeration written out in full: 'Article 7(2)'."""
    return f"{kind} {number}({paragraph})" if paragraph else f"{kind} {number}"


def divisions(text: str) -> list[Division]:
    """Article and annex mentions in order of appearance, enumerations expanded."""
    found: list[Division] = []
    for head in ARTICLE_REF.finditer(text):
        members = enumerated(head, text, ARTICLE_TAIL)
        found += [
            Division(
                start=m.start(),
                end=members[-1].end(),
                reference=Reference(
                    raw=cited("Article", m.group(1), m.group(2)),
                    article=m.group(1),
                    paragraph=m.group(2),
                ),
            )
            for m in members
        ]
    for head in ANNEX_REF.finditer(text):
        members = enumerated(head, text, ANNEX_TAIL)
        found += [
            Division(
                start=m.start(),
                end=members[-1].end(),
                reference=Reference(raw=cited("Annex", m.group(1)), annex=m.group(1)),
            )
            for m in members
        ]
    return sorted(found, key=lambda division: division.start)


def qualifies(text: str, end: int, span: Span) -> bool:
    """Whether an instrument at span is the one a division ending at end belongs to."""
    return span[0] >= end and QUALIFIER.match(text[end : span[0]]) is not None


def extract_references(text: str) -> tuple[Reference, ...]:
    """Structured cross-references found in the text, deduplicated."""
    mentions = instruments(text)
    found: list[Reference] = []
    attributed: set[Span] = set()

    for division in divisions(text):
        owner = next((pair for pair in mentions if qualifies(text, division.end, pair[0])), None)
        if owner is None:
            found.append(division.reference)
            continue
        (start, end), instrument = owner
        attributed.add((start, end))
        if instrument is None:
            continue
        raw = division.reference.raw + text[division.end : end]
        found.append(division.reference.model_copy(update={"raw": raw, "instrument": instrument}))

    found.extend(
        Reference(raw=text[start:end], instrument=instrument)
        for (start, end), instrument in mentions
        if (start, end) not in attributed and instrument is not None
    )
    return tuple(dict.fromkeys(found))
