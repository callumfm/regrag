"""Judge values: how a verdict becomes a score."""

from app.evals.judge.enums import JudgeVerdict
from app.evals.judge.models import (
    CaseJudgement,
    ClaimVerdict,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    RefusalVerdict,
)


def test_a_pass_scores_one_and_a_fail_zero() -> None:
    assert CorrectnessVerdict(critique="", verdict=JudgeVerdict.PASS).score() == 1.0
    assert CorrectnessVerdict(critique="", verdict=JudgeVerdict.FAIL).score() == 0.0
    assert RefusalVerdict(critique="", verdict=JudgeVerdict.PASS).score() == 1.0


def test_no_verdict_is_unmeasured_rather_than_a_fail() -> None:
    assert CorrectnessVerdict(critique="", verdict=JudgeVerdict.CANNOT_JUDGE).score() is None
    assert RefusalVerdict(critique="", verdict=JudgeVerdict.CANNOT_JUDGE).score() is None


def test_faithfulness_is_the_supported_share_of_the_claims() -> None:
    verdict = FaithfulnessVerdict(
        critique="",
        claims=(
            ClaimVerdict(claim="a", supported=True),
            ClaimVerdict(claim="b", supported=False),
            ClaimVerdict(claim="c", supported=True),
            ClaimVerdict(claim="d", supported=False),
        ),
    )

    assert verdict.score() == 0.5
    assert verdict.unsupported_claims() == ("b", "d")


def test_an_answer_making_no_claim_is_unmeasured_not_perfectly_faithful() -> None:
    assert FaithfulnessVerdict(critique="", claims=()).score() is None


def test_a_judgement_is_judged_when_any_dimension_came_back() -> None:
    assert not CaseJudgement().judged
    assert CaseJudgement(refusal=RefusalVerdict(critique="", verdict=JudgeVerdict.FAIL)).judged
