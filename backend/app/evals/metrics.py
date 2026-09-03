"""Eval scoring: what counts as a retrieved reference and a grounded citation, per case;
and the run's measures, each a plain function over its results."""

import re
from collections.abc import Sequence

from app.chat.enums import ChatNode, ChatOutcome
from app.chat.graph import assess_or_synthesize_or_refuse
from app.evals.dataset.enums import EvalKind
from app.evals.models import EvalMetrics, EvalResult, RetrievalMetrics
from app.retrieval.models import ReferenceTarget, RetrievedChunk

MARKER = re.compile(r"\[(\d+)\]")
"""A citation marker as the system prompt asks for it, like [1] or the [2][3] of a pair."""


def _division(item: ReferenceTarget | RetrievedChunk) -> tuple[str, str | None, str | None]:
    """The act and division a reference names: article case-folded, annex verbatim, and
    "" an unnumbered annex rather than none. Coarser than `follow`, which also matches the
    paragraph: a case names the article that answers it, so any chunk of that article
    counts as having found the reference."""
    article = item.article.lower() if item.article is not None else None
    return item.celex, article, item.annex


def score_reference_recall(
    targets: Sequence[ReferenceTarget], chunks: Sequence[RetrievedChunk]
) -> float:
    """Share of a case's authored references some retrieved chunk covers."""
    if not targets:
        return 0.0
    retrieved = {_division(chunk) for chunk in chunks}
    return sum(_division(target) in retrieved for target in targets) / len(targets)


def find_cited_markers(answer: str) -> tuple[int, ...]:
    """The distinct [n] markers the answer leans on, in the order it first cites them.
    Distinct, so citing one block repeatedly does not weight it by how often it is named."""
    seen = dict.fromkeys(int(match) for match in MARKER.findall(answer))
    return tuple(seen)


def score_citation_validity(answer: str, sources: Sequence[RetrievedChunk]) -> float | None:
    """Share of the answer's markers addressing a block it was given; None when it cited
    nothing, which is unmeasured rather than zero."""
    markers = find_cited_markers(answer)
    if not markers:
        return None
    return sum(1 <= marker <= len(sources) for marker in markers) / len(markers)


def score_reference_citation_rate(
    answer: str,
    sources: Sequence[RetrievedChunk],
    targets: Sequence[ReferenceTarget],
) -> float | None:
    """Share of a case's authored references the answer cited, scored over the references
    so an extra citation is not an error; None when the case names none."""
    if not targets:
        return None
    cited = {
        _division(sources[marker - 1])
        for marker in find_cited_markers(answer)
        if 1 <= marker <= len(sources)
    }
    return sum(_division(target) in cited for target in targets) / len(targets)


# Run metrics: each a plain function over the run's results, scoring the cases it applies to


def format_rate(value: float | None) -> str:
    """A rate as a figure, or a dash holding its place when it is unmeasured."""
    return f"{value:.2f}" if value is not None else "-"


def mean_or_none(values: Sequence[float | bool]) -> float | None:
    """The mean, or None when there is nothing to average — unmeasured, not zero."""
    return sum(values) / len(values) if values else None


def _scored(results: Sequence[EvalResult]) -> list[EvalResult]:
    """The cases that completed; one the graph raised on is counted but not scored."""
    return [r for r in results if r.state.error is None]


def scored_in_corpus(results: Sequence[EvalResult]) -> list[EvalResult]:
    return [r for r in _scored(results) if r.case.kind is EvalKind.IN_CORPUS]


def scored_out_of_corpus(results: Sequence[EvalResult]) -> list[EvalResult]:
    return [r for r in _scored(results) if r.case.kind is EvalKind.OUT_OF_CORPUS]


def _raw_recall(result: EvalResult) -> float:
    return score_reference_recall(result.case.references, result.state.hits)


def _expanded_recall(result: EvalResult) -> float:
    return score_reference_recall(result.case.references, result.state.sources)


def _gate_refused(result: EvalResult) -> bool:
    """The branch the graph took, observed off the path so a routing bug shows up here.
    A retrieval-only run never routes, so its gate is read the way the graph would route."""
    if result.state.outcome is ChatOutcome.ABORTED:
        return assess_or_synthesize_or_refuse(result.state) is ChatNode.REFUSE
    return result.state.outcome is ChatOutcome.REFUSED


def count_errors(results: Sequence[EvalResult]) -> int:
    return len(results) - len(_scored(results))


def count_cases_of_kind(results: Sequence[EvalResult], kind: EvalKind) -> int:
    """Cases of this kind the run covered, errored or not: the dataset's shape, which a
    case the graph raised on does not change."""
    return sum(result.case.kind is kind for result in results)


def compute_raw_hit_rate(results: Sequence[EvalResult]) -> float | None:
    """Share of in-corpus cases where search found at least one authored reference."""
    return mean_or_none([_raw_recall(r) > 0 for r in scored_in_corpus(results)])


def compute_raw_recall(results: Sequence[EvalResult]) -> float | None:
    """Mean share of authored references search found, before expansion widened it."""
    return mean_or_none([_raw_recall(r) for r in scored_in_corpus(results)])


def compute_expanded_hit_rate(results: Sequence[EvalResult]) -> float | None:
    """Share of in-corpus cases where at least one authored reference reached the prompt."""
    return mean_or_none([_expanded_recall(r) > 0 for r in scored_in_corpus(results)])


def compute_expanded_recall(results: Sequence[EvalResult]) -> float | None:
    """Mean share of authored references that reached the prompt."""
    return mean_or_none([_expanded_recall(r) for r in scored_in_corpus(results)])


def compute_cited_references(results: Sequence[EvalResult]) -> float | None:
    """Mean share of authored references the answers cited, over the in-corpus cases."""
    rates = [
        score_reference_citation_rate(r.state.answer, r.state.sources, r.case.references)
        for r in scored_in_corpus(results)
    ]
    return mean_or_none([rate for rate in rates if rate is not None])


def compute_markers_in_context(results: Sequence[EvalResult]) -> float | None:
    """Mean share of markers addressing a block that was in context, over the answers
    that cited anything."""
    validities = [
        score_citation_validity(r.state.answer, r.state.sources) for r in _scored(results)
    ]
    return mean_or_none([v for v in validities if v is not None])


def _judged(results: Sequence[EvalResult]) -> list[EvalResult]:
    """The scored cases the judge returned a verdict on."""
    return [r for r in _scored(results) if r.judgement is not None and r.judgement.judged]


def count_judged(results: Sequence[EvalResult]) -> int:
    return len(_judged(results))


def compute_correctness(results: Sequence[EvalResult]) -> float | None:
    """Share of judged in-corpus answers the judge passed against the reference answer."""
    scores = [
        r.judgement.correctness.score()
        for r in _judged(results)
        if r.judgement is not None and r.judgement.correctness is not None
    ]
    return mean_or_none([s for s in scores if s is not None])


def compute_faithfulness(results: Sequence[EvalResult]) -> float | None:
    """Mean share of an answer's claims its cited context backs, over the judged answers
    that made a checkable claim."""
    scores = [
        r.judgement.faithfulness.score()
        for r in _judged(results)
        if r.judgement is not None and r.judgement.faithfulness is not None
    ]
    return mean_or_none([s for s in scores if s is not None])


def compute_model_refusal_rate(results: Sequence[EvalResult]) -> float | None:
    """Share of judged out-of-corpus answers that declined in the model's own words."""
    scores = [
        r.judgement.refusal.score()
        for r in _judged(results)
        if r.judgement is not None and r.judgement.refusal is not None
    ]
    return mean_or_none([s for s in scores if s is not None])


def compute_gate_refusal_rate(results: Sequence[EvalResult]) -> float | None:
    """Share of out-of-corpus cases the pre-model gate refused."""
    return mean_or_none([_gate_refused(r) for r in scored_out_of_corpus(results)])


def count_false_refusals(results: Sequence[EvalResult]) -> int:
    """In-corpus cases the gate refused."""
    return sum(_gate_refused(r) for r in scored_in_corpus(results))


def count_refusals_of_a_found_reference(results: Sequence[EvalResult]) -> int:
    """In-corpus cases refused though search had found an authored reference: the gate too
    tight, rather than a corpus that does not cover the question."""
    return sum(_gate_refused(r) and _raw_recall(r) > 0 for r in scored_in_corpus(results))


def compute_mean_step_ms(results: Sequence[EvalResult]) -> dict[str, int]:
    """Each step's mean time over the cases that ran it, in the order steps first appear."""
    timings: dict[str, list[int]] = {}
    for result in _scored(results):
        for step in result.state.steps:
            timings.setdefault(step.step.value, []).append(step.ms)
    return {name: sum(ms) // len(ms) for name, ms in timings.items()}


def compute_mean_total_ms(results: Sequence[EvalResult]) -> int:
    return int(mean_or_none([r.state.total_ms or 0 for r in _scored(results)]) or 0)


def compute_input_tokens(results: Sequence[EvalResult]) -> int:
    return sum(r.state.token_totals()[0] or 0 for r in _scored(results))


def compute_output_tokens(results: Sequence[EvalResult]) -> int:
    return sum(r.state.token_totals()[1] or 0 for r in _scored(results))


def compute_retrieval_metrics(results: Sequence[EvalResult]) -> RetrievalMetrics:
    """The retrieval measures every run kind shares, each over the cases it applies to."""
    return RetrievalMetrics(
        cases=len(results),
        in_corpus=count_cases_of_kind(results, EvalKind.IN_CORPUS),
        out_of_corpus=count_cases_of_kind(results, EvalKind.OUT_OF_CORPUS),
        errors=count_errors(results),
        raw_hit_rate=compute_raw_hit_rate(results),
        raw_recall=compute_raw_recall(results),
        expanded_hit_rate=compute_expanded_hit_rate(results),
        expanded_recall=compute_expanded_recall(results),
        gate_refusal_rate=compute_gate_refusal_rate(results),
        false_refusals=count_false_refusals(results),
        refused_a_found_reference=count_refusals_of_a_found_reference(results),
    )


def compute_metrics(results: Sequence[EvalResult]) -> EvalMetrics:
    """Every measure of the run: the shared retrieval block plus what the model calls added."""
    return EvalMetrics(
        **compute_retrieval_metrics(results).model_dump(),
        cited_references=compute_cited_references(results),
        markers_in_context=compute_markers_in_context(results),
        correctness=compute_correctness(results),
        faithfulness=compute_faithfulness(results),
        model_refusal_rate=compute_model_refusal_rate(results),
        judged=count_judged(results),
        mean_step_ms=compute_mean_step_ms(results),
        mean_total_ms=compute_mean_total_ms(results),
        input_tokens=compute_input_tokens(results),
        output_tokens=compute_output_tokens(results),
    )
