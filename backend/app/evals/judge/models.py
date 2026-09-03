"""Judge values: what each dimension's verdict looks like, and how a verdict becomes a score.

Each verdict is written with its critique first: the judge answers in field order, so the
reasoning is on the page before the verdict is decided, and is there to hand-check against.
"""

from app.core.models import FrozenModel
from app.evals.judge.enums import CorrectnessFailure, JudgeVerdict


def _score_verdict(verdict: JudgeVerdict) -> float | None:
    """A pass as 1, a fail as 0, and no verdict as unmeasured."""
    if verdict is JudgeVerdict.CANNOT_JUDGE:
        return None
    return float(verdict is JudgeVerdict.PASS)


class CorrectnessVerdict(FrozenModel):
    """Whether the answer states what the reference answer states: the same rule, figures
    and act, not the same words. A fail names its kind."""

    critique: str
    verdict: JudgeVerdict
    failure: CorrectnessFailure | None = None

    def score(self) -> float | None:
        return _score_verdict(self.verdict)


class ClaimVerdict(FrozenModel):
    """One factual claim the answer makes, and whether a cited block backs it."""

    claim: str
    supported: bool


class FaithfulnessVerdict(FrozenModel):
    """The answer's claims, each checked against the context it cited. An answer making no
    checkable claim is unmeasured rather than perfectly faithful."""

    critique: str
    claims: tuple[ClaimVerdict, ...]

    def score(self) -> float | None:
        if not self.claims:
            return None
        return sum(claim.supported for claim in self.claims) / len(self.claims)

    def unsupported_claims(self) -> tuple[str, ...]:
        return tuple(claim.claim for claim in self.claims if not claim.supported)


class RefusalVerdict(FrozenModel):
    """Whether an answer to a question the corpus does not cover declined, in the model's
    own words, rather than answering from memory. A pass is a decline."""

    critique: str
    verdict: JudgeVerdict

    def score(self) -> float | None:
        return _score_verdict(self.verdict)


class CaseJudgement(FrozenModel):
    """What the judge said about one case: the dimensions that applied to it, each None
    when it did not apply or the judge call failed."""

    correctness: CorrectnessVerdict | None = None
    faithfulness: FaithfulnessVerdict | None = None
    refusal: RefusalVerdict | None = None

    @property
    def judged(self) -> bool:
        """Whether any dimension came back with a verdict."""
        return any(v is not None for v in (self.correctness, self.faithfulness, self.refusal))
