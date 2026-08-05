"""CELEX ids: the EU's identity scheme for legal acts, read and written in one place."""

from app.core.clock import utc_today

LEGISLATION = "3"
CONSOLIDATED = "0"
CENTURIES = ("19", "20")
LENGTH = 10
KIND_LETTERS = {"regulation": "R", "directive": "L", "decision": "D"}


def expand_year(value: str) -> str:
    """Two-digit years in older citations are 20th century: '92' -> '1992'."""
    return f"19{value}" if len(value) == 2 else value


def as_year(value: str) -> int | None:
    """The year a citation fragment denotes, or None if it can only be an act number."""
    year = expand_year(value)
    if len(year) != 4 or year[:2] not in CENTURIES or int(year) > utc_today().year:
        return None
    return int(year)


def build(kind: str, year: str, number: str) -> str:
    """A legislation id: sector, year, kind letter, then zero-padded number."""
    ref = f"{LEGISLATION}{expand_year(year)}{KIND_LETTERS[kind.lower()]}{number.zfill(4)}"
    if as_year(year) is None or not is_legislation(ref):
        raise ValueError(f"not a legislation citation: {kind} {number}/{year}")
    return ref


def is_legislation(ref: str) -> bool:
    """Whether an id names a regulation, directive or decision."""
    return ref.startswith(LEGISLATION) and len(ref) == LENGTH and ref[5] in KIND_LETTERS.values()


def consolidated_stem(ref: str) -> str:
    """Prefix shared by every consolidated version of an act: 32015R0757 -> 02015R0757-."""
    return f"{CONSOLIDATED}{ref[1:]}-"
