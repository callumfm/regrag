"""Each eval case as one line: what search found, what reached the prompt, what was cited,
what the judge made of it — and, under a case the judge failed, why."""

from collections.abc import Sequence

from app.evals.judge.enums import JudgeVerdict
from app.evals.judge.models import CaseJudgement
from app.evals.metrics import score_reference_citation_rate, score_reference_recall
from app.evals.models import EvalResult

INDENT = "    "
UNMEASURED = "-"
"""Holds a figure's place when no case measured it."""


def format_rate(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else UNMEASURED


def _format_case_line(result: EvalResult, width: int) -> str:
    """One case on one line: what search found, what reached the prompt, what the answer
    cited of the references the case authors, what the judge scored, then how the run ended."""
    state, references, judgement = result.state, result.case.references, result.judgement
    scored = state.error is None
    recalled = scored and bool(references)
    raw = score_reference_recall(references, state.hits) if recalled else None
    expanded = score_reference_recall(references, state.sources) if recalled else None
    cited = (
        score_reference_citation_rate(state.answer, state.sources, references) if scored else None
    )
    correctness = judgement.correctness.score() if judgement and judgement.correctness else None
    faithfulness = judgement.faithfulness.score() if judgement and judgement.faithfulness else None
    return (
        f"{result.case.id:<{width}}  raw {format_rate(raw):>4}  exp {format_rate(expanded):>4}  "
        f"cite {format_rate(cited):>4}  corr {format_rate(correctness):>4}  "
        f"faith {format_rate(faithfulness):>4}  {state.outcome.value:<8}{state.total_ms or 0:>6}ms"
        f"{'  ' + state.error if state.error else ''}"
    )


def _format_critiques(judgement: CaseJudgement) -> list[str]:
    """The judge's reasons, for the dimensions it did not pass: a pass stays on the case
    line, so the report reads as one line per case until something needs reading."""
    lines = []
    correctness, faithfulness, refusal = (
        judgement.correctness,
        judgement.faithfulness,
        judgement.refusal,
    )
    if correctness and correctness.verdict is not JudgeVerdict.PASS:
        failure = f" ({correctness.failure.value})" if correctness.failure else ""
        lines.append(f"correctness {correctness.verdict.value}{failure}: {correctness.critique}")
    if faithfulness and (unsupported := faithfulness.unsupported_claims()):
        lines.append(f"faithfulness {format_rate(faithfulness.score())}: {faithfulness.critique}")
        lines.extend(f"unsupported: {claim}" for claim in unsupported)
    if refusal and refusal.verdict is not JudgeVerdict.PASS:
        lines.append(f"refusal {refusal.verdict.value}: {refusal.critique}")
    return [INDENT + line for line in lines]


def format_case_lines(results: Sequence[EvalResult]) -> list[str]:
    """Every case as its own line, the id column sized to the longest id in the run, with
    the judge's critiques under any case it did not pass. A case that raised scores nothing,
    as the aggregate leaves it out; one authoring no reference has no recall to measure,
    and prints a dash rather than a zero."""
    if not results:
        return []
    width = max(len(result.case.id) for result in results)
    lines = []
    for result in results:
        lines.append(_format_case_line(result, width))
        if result.judgement is not None:
            lines.extend(_format_critiques(result.judgement))
    return lines
