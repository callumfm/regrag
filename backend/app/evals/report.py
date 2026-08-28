"""Each eval case as one line: what search found, what reached the prompt, what was cited."""

from collections.abc import Sequence

from app.evals.metrics import format_rate, score_reference_citation_rate, score_reference_recall
from app.evals.models import EvalResult


def _format_case_line(result: EvalResult, width: int) -> str:
    """One case on one line: what search found, what reached the prompt, what the answer
    cited of the references the case authors, then how the run ended."""
    state, references = result.state, result.case.references
    scored = state.error is None
    recalled = scored and bool(references)
    raw = score_reference_recall(references, state.hits) if recalled else None
    expanded = score_reference_recall(references, state.sources) if recalled else None
    cited = (
        score_reference_citation_rate(state.answer, state.sources, references) if scored else None
    )
    return (
        f"{result.case.id:<{width}}  raw {format_rate(raw):>4}  exp {format_rate(expanded):>4}  "
        f"cite {format_rate(cited):>4}  {state.outcome.value:<8}{state.total_ms or 0:>6}ms"
        f"{'  ' + state.error if state.error else ''}"
    )


def format_case_lines(results: Sequence[EvalResult]) -> list[str]:
    """Every case as its own line, the id column sized to the longest id in the run. A case
    that raised scores nothing, as the aggregate leaves it out; one authoring no reference
    has no recall to measure, and prints a dash rather than a zero."""
    if not results:
        return []
    width = max(len(result.case.id) for result in results)
    return [_format_case_line(result, width) for result in results]
