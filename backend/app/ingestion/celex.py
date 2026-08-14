"""CELEX ids: the EU's identity scheme for legal acts, read and written in one place."""

from app.core.clock import utc_today

LEGISLATION = "3"
CONSOLIDATED = "0"
CENTURIES = ("19", "20")
LENGTH = 10
KIND_LETTERS = {"regulation": "R", "directive": "L", "decision": "D"}
YEAR_FIRST_SCHEME = 2015
"""From this year on acts are numbered year-first, and stopped carrying a 'No'."""


def expand_year(value: str) -> str:
    """Two-digit years in older citations are 20th century: '92' -> '1992'."""
    return f"19{value}" if len(value) == 2 else value


def as_year(value: str) -> int | None:
    """The year a citation fragment denotes, or None if it can only be an act number."""
    year = expand_year(value)
    if len(year) != 4 or year[:2] not in CENTURIES or int(year) > utc_today().year:
        return None
    return int(year)


def order_number_and_year(numbered: bool, first: str, second: str) -> tuple[str, str]:
    """A '765/2008' pair as (number, year); the 'No NN/MM' form is number-first.

    Acts numbered under the 2015 year-first scheme never carried a 'No', so a leading
    four-digit year from that scheme marks a sloppy citation to read year-first anyway.
    """
    if numbered:
        year = as_year(first)
        if len(first) == 4 and year is not None and year >= YEAR_FIRST_SCHEME:
            return second, first
        return first, second
    if len(first) == 2 and len(second) == 2:
        return second, first
    left, right = as_year(first), as_year(second)
    if left and right:
        return (second, first) if left >= right else (first, second)
    if right:
        return first, second
    return second, first


def build(kind: str, year: str, number: str) -> str:
    """A legislation id: sector, year, kind letter, then zero-padded number."""
    celex = f"{LEGISLATION}{expand_year(year)}{KIND_LETTERS[kind.lower()]}{number.zfill(4)}"
    if as_year(year) is None or not is_legislation(celex):
        raise ValueError(f"not a legislation citation: {kind} {number}/{year}")
    return celex


def is_legislation(celex: str) -> bool:
    """Whether an id names a regulation, directive or decision."""
    return (
        celex.startswith(LEGISLATION) and len(celex) == LENGTH and celex[5] in KIND_LETTERS.values()
    )


def consolidated_stem(celex: str) -> str:
    """Prefix shared by every consolidated version of an act: 32015R0757 -> 02015R0757-."""
    return f"{CONSOLIDATED}{celex[1:]}-"
