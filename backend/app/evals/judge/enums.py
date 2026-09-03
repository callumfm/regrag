"""Judge enumerations."""

from enum import StrEnum


class JudgeVerdict(StrEnum):
    """What a judged dimension came to: a pass, a fail, or no verdict at all.

    CANNOT_JUDGE is the way out: a case the material does not settle scores as unmeasured
    rather than forcing a verdict the judge would have to invent.
    """

    PASS = "pass"
    FAIL = "fail"
    CANNOT_JUDGE = "cannot_judge"


class CorrectnessFailure(StrEnum):
    """Why a correct-looking answer failed, so failures can be counted by kind rather than
    read one by one. Grown from observed failures; OTHER is the one for a new kind."""

    MISSING_FACT = "missing_fact"
    WRONG_FIGURE = "wrong_figure"
    CONTRADICTS_REFERENCE = "contradicts_reference"
    INVENTED_NAME = "invented_name"
    OTHER = "other"
