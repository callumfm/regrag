"""Eval enumerations."""

from enum import StrEnum


class EvalKind(StrEnum):
    """What a case tests: that the corpus answers it, or that the graph refuses it."""

    IN_CORPUS = "in_corpus"
    OUT_OF_CORPUS = "out_of_corpus"
