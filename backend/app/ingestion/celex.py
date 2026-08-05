"""CELEX ids: the EU's identity scheme for legal acts, read and written in one place."""

LEGISLATION = "3"
CONSOLIDATED = "0"
KIND_LETTERS = {"regulation": "R", "directive": "L", "decision": "D"}


def build(kind: str, year: str, number: str) -> str:
    """A legislation id: sector, year, kind letter, then zero-padded number."""
    return f"{LEGISLATION}{year}{KIND_LETTERS[kind.lower()]}{number.zfill(4)}"


def is_legislation(ref: str) -> bool:
    """Whether an id names a regulation, directive or decision."""
    return ref.startswith(LEGISLATION) and len(ref) > 5 and ref[5] in KIND_LETTERS.values()


def consolidated_stem(ref: str) -> str:
    """Prefix shared by every consolidated version of an act: 32015R0757 -> 02015R0757-."""
    return f"{CONSOLIDATED}{ref[1:]}-"
