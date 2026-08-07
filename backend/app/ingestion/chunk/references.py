"""Cross-reference extraction: find the mentions, then attribute each division to its instrument.

"Article 7 of Regulation X" is two mentions, not one: a division ("Article 7") and the
instrument that qualifies it. Divisions no instrument qualifies belong to this document.
"""

import re
from collections.abc import Sequence

from app.core.models import FrozenModel
from app.ingestion import celex
from app.ingestion.chunk.models import Reference

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


class Mention(FrozenModel):
    """Where in the text one reference was found."""

    start: int
    end: int

    def is_qualified_by(self, other: "Mention", text: str) -> bool:
        """True when nothing but a qualifier ('of', 'to', 'in') separates this mention from it."""
        return other.start >= self.end and QUALIFIER.match(text[self.end : other.start]) is not None


class Division(Mention):
    """One member of an article or annex mention, ending where its enumeration does."""

    reference: Reference


class InstrumentMention(Mention):
    """One cited instrument; celex is None where the citation cannot resolve to an id."""

    celex: str | None


def order_number_and_year(numbered: bool, first: str, second: str) -> tuple[str, str]:
    """A '765/2008' pair as (number, year); EU drafting uses both orderings."""
    left, right = celex.as_year(first), celex.as_year(second)
    if left and right:
        return (second, first) if left >= right else (first, second)
    if right:
        return first, second
    if left:
        return second, first
    return (first, second) if numbered else (second, first)


def expand_enumeration(
    head: re.Match[str], text: str, tail: re.Pattern[str]
) -> list[re.Match[str]]:
    """An 'Articles 6, 7 and 8' run as one match per member."""
    members = [head]
    while member := tail.match(text, members[-1].end()):
        members.append(member)
    return members


def format_citation(kind: str, number: str, paragraph: str | None = None) -> str:
    """One member of an enumeration written out in full: 'Article 7(2)'."""
    return f"{kind} {number}({paragraph})" if paragraph else f"{kind} {number}"


def find_instrument_mentions(text: str) -> list[InstrumentMention]:
    """Every instrument mention in order of appearance."""
    found: list[InstrumentMention] = []
    for match in INSTRUMENT_REF.finditer(text):
        number, year = order_number_and_year(bool(match.group(2)), match.group(3), match.group(4))
        try:
            resolved = celex.build(match.group(1), year, number)
        except ValueError:
            resolved = None
        start, end = match.span()
        found.append(InstrumentMention(start=start, end=end, celex=resolved))
    return found


def find_division_mentions(text: str) -> list[Division]:
    """Article and annex mentions in order of appearance, enumerations expanded."""
    found: list[Division] = []
    for head in ARTICLE_REF.finditer(text):
        members = expand_enumeration(head, text, ARTICLE_TAIL)
        found += [
            Division(
                start=m.start(),
                end=members[-1].end(),
                reference=Reference(
                    raw=format_citation("Article", m.group(1), m.group(2)),
                    article=m.group(1),
                    paragraph=m.group(2),
                ),
            )
            for m in members
        ]
    for head in ANNEX_REF.finditer(text):
        members = expand_enumeration(head, text, ANNEX_TAIL)
        found += [
            Division(
                start=m.start(),
                end=members[-1].end(),
                reference=Reference(raw=format_citation("Annex", m.group(1)), annex=m.group(1)),
            )
            for m in members
        ]
    return sorted(found, key=lambda division: division.start)


def attribute_divisions_to_instruments(
    text: str, divisions: Sequence[Division], instruments: Sequence[InstrumentMention]
) -> list[Reference]:
    """Each division as a reference, carrying the instrument that qualifies it, if any.

    A division qualified by an unresolvable instrument is dropped: it is not this document's.
    """
    found: list[Reference] = []
    for division in divisions:
        owner = next((m for m in instruments if division.is_qualified_by(m, text)), None)
        if owner is None:
            found.append(division.reference)
        elif owner.celex is not None:
            raw = division.reference.raw + text[division.end : owner.end]
            found.append(
                division.reference.model_copy(update={"raw": raw, "instrument": owner.celex})
            )
    return found


def unattributed_instruments(
    text: str, divisions: Sequence[Division], instruments: Sequence[InstrumentMention]
) -> list[Reference]:
    """Instruments cited in their own right: the ones no division was attributed to."""
    return [
        Reference(raw=text[m.start : m.end], instrument=m.celex)
        for m in instruments
        if m.celex is not None
        and not any(division.is_qualified_by(m, text) for division in divisions)
    ]


def extract_references(text: str) -> tuple[Reference, ...]:
    """Structured cross-references found in the text, deduplicated."""
    divisions = find_division_mentions(text)
    instruments = find_instrument_mentions(text)
    return tuple(
        dict.fromkeys(
            attribute_divisions_to_instruments(text, divisions, instruments)
            + unattributed_instruments(text, divisions, instruments)
        )
    )
