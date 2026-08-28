"""Eval run values: one scored case, and what a whole run measured."""

from typing import Any

from app.chat.models import ChatState
from app.core.models import FrozenModel
from app.evals.dataset.models import EvalCase


class EvalResult(FrozenModel):
    """One case driven through the chat graph: the case, and the run it produced — the
    same state a chat request ends in, so a run is scored off what production records."""

    case: EvalCase
    state: ChatState


class RetrievalMetrics(FrozenModel):
    """The measures retrieval alone fills. A rate is None when no case measured it.

    in_corpus / out_of_corpus: how the run's cases were authored, counted whether or not
        they scored, so errors overlaps them rather than partitioning with them.
    raw_*: scored on what search found; expanded_*: on what reached the prompt.
    gate_refusal_rate: out-of-corpus cases the pre-model gate refused; a model declining in
        its own words is the judge's to score.
    false_refusals: in-corpus cases the gate refused; refused_a_found_reference: those where
        search had already found an authored reference — the gate too tight.
    """

    cases: int
    in_corpus: int
    out_of_corpus: int
    errors: int
    raw_hit_rate: float | None
    raw_recall: float | None
    expanded_hit_rate: float | None
    expanded_recall: float | None
    gate_refusal_rate: float | None
    false_refusals: int
    refused_a_found_reference: int


class EvalMetrics(RetrievalMetrics):
    """The retrieval measures plus what the model calls added.

    cited_references: share of authored references the answers cited.
    markers_in_context: share of [n] markers addressing a block that was in context.
    mean_node_ms: a node's mean time over the cases that ran it.
    """

    cited_references: float | None
    markers_in_context: float | None
    mean_node_ms: dict[str, int]
    mean_total_ms: int
    input_tokens: int
    output_tokens: int


class EvalRun(FrozenModel):
    """One eval run: which dataset and settings it scored, what it measured, and every case.

    dataset_sha hashes what the cases assert; case_filter names the subset actually scored.
    corpus_version names the ingest the corpus stands at, so two scores are only compared
    when they were measured against the same text; stale_cases names the cases whose cited
    text has moved since they were authored, whose reference answers are owed a re-review.
    cached says the run had the call cache on, so an embed or rerank timing may measure a
    disk read rather than the provider — a cached run is not a latency baseline.
    """

    dataset_sha: str
    case_filter: str | None = None
    corpus_version: str | None = None
    stale_cases: tuple[str, ...] = ()
    cached: bool = False
    settings: dict[str, Any]
    metrics: EvalMetrics
    results: tuple[EvalResult, ...]

    def summary(self) -> str:
        """The run's setup and scores as JSON, then any case owed a re-review, then any case
        the graph raised on. A stale case is reported, never failed: only a human can repair
        one, so the run stays green and says what needs reading."""
        blocks = [
            self.model_dump_json(
                indent=2,
                include={
                    "dataset_sha",
                    "case_filter",
                    "corpus_version",
                    "cached",
                    "settings",
                    "metrics",
                },
            )
        ]
        if self.stale_cases:
            count = len(self.stale_cases)
            subject = "case cites" if count == 1 else "cases cite"
            blocks.append(
                f"{count} {subject} text that changed since authoring:\n"
                + "\n".join(f"  {case_id}" for case_id in self.stale_cases)
            )
        errored = [f"  {r.case.id}  {r.state.error}" for r in self.results if r.state.error]
        if errored:
            blocks.append("\n".join(["errored:", *errored]))
        return "\n\n".join(blocks)
