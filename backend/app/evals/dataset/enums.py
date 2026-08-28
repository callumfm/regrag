"""Dataset enumerations."""

from enum import StrEnum


class EvalKind(StrEnum):
    """What a case tests: that the corpus answers it, or that the graph refuses it."""

    IN_CORPUS = "in_corpus"
    OUT_OF_CORPUS = "out_of_corpus"


class DriftKind(StrEnum):
    """How far a case reference has come adrift of the corpus it was authored against.

    UNRESOLVED is hard drift, which a run scores as a retrieval miss; STALE is soft drift,
    which a run scores green against an answer that may no longer be right; UNSTAMPED is
    neither, but it is a reference drift cannot be seen on at all.
    """

    UNRESOLVED = "unresolved"
    STALE = "stale"
    UNSTAMPED = "unstamped"
