"""What one eval run produced: a result per case, and the run's aggregate scores."""

from collections.abc import Sequence

from app.chat.enums import ChatNode
from app.core.config import config
from app.core.models import FrozenModel
from app.evals.enums import EvalKind
from app.evals.metrics import (
    score_citation_validity,
    score_reference_citation_rate,
    score_reference_recall,
)
from app.evals.models import EvalCase
from app.retrieval.models import RetrievedChunk, SearchResult


def _mean_or_none(values: Sequence[float | bool]) -> float | None:
    """The mean, or None when there is nothing to average — unmeasured, not zero."""
    return sum(values) / len(values) if values else None


class RunSettings(FrozenModel):
    """Every knob that moves a hit, so two runs are only compared when comparable; the
    reranker's model and threshold are None when it did not run."""

    chat_model: str
    chat_sources: int
    chat_context_chunks: int
    expand_sections: bool
    embed_model: str
    search_candidates: int
    rrf_k: int
    rerank_enabled: bool
    rerank_model: str | None
    min_cosine_similarity: float
    min_reranker_relevance: float | None

    @classmethod
    def from_config(cls) -> "RunSettings":
        """The settings as the run will really read them."""
        reranking = config.RERANK_ENABLED
        return cls(
            chat_model=config.CHAT_MODEL,
            chat_sources=config.CHAT_SOURCES,
            chat_context_chunks=config.CHAT_CONTEXT_CHUNKS,
            expand_sections=config.EXPAND_SECTIONS,
            embed_model=config.EMBED_MODEL,
            search_candidates=config.SEARCH_CANDIDATES,
            rrf_k=config.RRF_K,
            rerank_enabled=reranking,
            rerank_model=config.RERANK_MODEL if reranking else None,
            min_cosine_similarity=config.MIN_COSINE_SIMILARITY,
            min_reranker_relevance=config.MIN_RERANKER_RELEVANCE if reranking else None,
        )


class CaseResult(FrozenModel):
    """One case driven through the graph: what it retrieved, what it answered, what it cost.

    nodes: the graph nodes that ran, in order, which is what a refusal is read from.
    hits: the raw search results, before the gate and before expansion.
    error: what the graph raised; such a case is counted but not scored.
    """

    case: EvalCase
    nodes: tuple[ChatNode, ...] = ()
    hits: tuple[SearchResult, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()
    answer: str = ""
    retrieve_ms: int = 0
    total_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    @property
    def synthesize_ms(self) -> int:
        """What the model call cost, once retrieval had already run."""
        return self.total_ms - self.retrieve_ms

    @property
    def raw_recall(self) -> float:
        """Share of authored references search itself found, before expansion widened it."""
        return score_reference_recall(self.case.references, self.hits)

    @property
    def expanded_recall(self) -> float:
        """Share of authored references that reached the prompt."""
        return score_reference_recall(self.case.references, self.sources)

    @property
    def citation_validity(self) -> float | None:
        """Share of the answer's markers that address a block it was actually given."""
        return score_citation_validity(self.answer, self.sources)

    @property
    def reference_citation_rate(self) -> float | None:
        """Share of the case's authored references the answer actually cited."""
        return score_reference_citation_rate(self.answer, self.sources, self.case.references)

    @property
    def gate_refused(self) -> bool:
        """Whether the graph ended at the refusal node, which is the gate firing."""
        return bool(self.nodes) and self.nodes[-1] is ChatNode.REFUSE

    @property
    def refused_a_covered_case(self) -> bool:
        """An in-corpus case refused though search had found a reference: the gate too
        tight, rather than a corpus that does not cover the question."""
        return self.case.kind is EvalKind.IN_CORPUS and self.gate_refused and self.raw_recall > 0


class RetrievalMetrics(FrozenModel):
    """What search found, scored before expansion and again after it."""

    raw_hit_rate: float | None
    raw_recall: float | None
    expanded_hit_rate: float | None
    expanded_recall: float | None

    @classmethod
    def from_cases(cls, in_corpus: Sequence[CaseResult]) -> "RetrievalMetrics":
        return cls(
            raw_hit_rate=_mean_or_none([r.raw_recall > 0 for r in in_corpus]),
            raw_recall=_mean_or_none([r.raw_recall for r in in_corpus]),
            expanded_hit_rate=_mean_or_none([r.expanded_recall > 0 for r in in_corpus]),
            expanded_recall=_mean_or_none([r.expanded_recall for r in in_corpus]),
        )


class CitationMetrics(FrozenModel):
    """How the answers cited: which authored references they leaned on, and whether every
    marker addressed a block that was really in context."""

    cited_references: float | None
    markers_in_context: float | None

    @classmethod
    def from_cases(
        cls, in_corpus: Sequence[CaseResult], scored: Sequence[CaseResult]
    ) -> "CitationMetrics":
        rates = [
            r.reference_citation_rate for r in in_corpus if r.reference_citation_rate is not None
        ]
        validities = [r.citation_validity for r in scored if r.citation_validity is not None]
        return cls(
            cited_references=_mean_or_none(rates),
            markers_in_context=_mean_or_none(validities),
        )


class RefusalMetrics(FrozenModel):
    """The pre-model gate only; a model declining in its own words is the judge's to score.

    false_refusals: in-corpus cases the gate refused.
    over_a_found_reference: those where search had already found an authored reference.
    """

    gate_rate: float | None
    false_refusals: int
    over_a_found_reference: int

    @classmethod
    def from_cases(
        cls, in_corpus: Sequence[CaseResult], out_of_corpus: Sequence[CaseResult]
    ) -> "RefusalMetrics":
        return cls(
            gate_rate=_mean_or_none([r.gate_refused for r in out_of_corpus]),
            false_refusals=sum(r.gate_refused for r in in_corpus),
            over_a_found_reference=sum(r.refused_a_covered_case for r in in_corpus),
        )


class LatencyMetrics(FrozenModel):
    """How long a case took, averaged over the cases that completed."""

    mean_retrieve_ms: int
    mean_total_ms: int

    @classmethod
    def from_cases(cls, scored: Sequence[CaseResult]) -> "LatencyMetrics":
        return cls(
            mean_retrieve_ms=int(_mean_or_none([r.retrieve_ms for r in scored]) or 0),
            mean_total_ms=int(_mean_or_none([r.total_ms for r in scored]) or 0),
        )


class UsageMetrics(FrozenModel):
    """What the run's answers cost, summed over the cases that completed."""

    input_tokens: int
    output_tokens: int

    @classmethod
    def from_cases(cls, scored: Sequence[CaseResult]) -> "UsageMetrics":
        return cls(
            input_tokens=sum(r.input_tokens or 0 for r in scored),
            output_tokens=sum(r.output_tokens or 0 for r in scored),
        )


class RunMetrics(FrozenModel):
    """What a run measured, grouped by what each group answers about."""

    cases: int
    in_corpus: int
    out_of_corpus: int
    errors: int
    retrieval: RetrievalMetrics
    citations: CitationMetrics
    refusals: RefusalMetrics
    latency: LatencyMetrics
    usage: UsageMetrics

    @classmethod
    def from_cases(cls, results: Sequence[CaseResult]) -> "RunMetrics":
        """Aggregate the completed cases; an errored case counts toward `errors` only."""
        scored = [r for r in results if r.error is None]
        in_corpus = [r for r in scored if r.case.kind is EvalKind.IN_CORPUS]
        out_of_corpus = [r for r in scored if r.case.kind is EvalKind.OUT_OF_CORPUS]
        return cls(
            cases=len(results),
            in_corpus=len(in_corpus),
            out_of_corpus=len(out_of_corpus),
            errors=len(results) - len(scored),
            retrieval=RetrievalMetrics.from_cases(in_corpus),
            citations=CitationMetrics.from_cases(in_corpus, scored),
            refusals=RefusalMetrics.from_cases(in_corpus, out_of_corpus),
            latency=LatencyMetrics.from_cases(scored),
            usage=UsageMetrics.from_cases(scored),
        )


class RunResult(FrozenModel):
    """One eval run: which dataset and settings it scored, and every case.

    dataset_sha hashes the whole dataset; case_pattern names the subset actually scored.
    """

    dataset_sha: str
    case_pattern: str | None = None
    settings: RunSettings
    metrics: RunMetrics
    results: tuple[CaseResult, ...]

    @classmethod
    def from_results(
        cls,
        results: Sequence[CaseResult],
        dataset_sha: str,
        case_pattern: str | None = None,
    ) -> "RunResult":
        """A finished run, with its cases aggregated and the settings they ran under."""
        return cls(
            dataset_sha=dataset_sha,
            case_pattern=case_pattern,
            settings=RunSettings.from_config(),
            metrics=RunMetrics.from_cases(results),
            results=tuple(results),
        )

    def summary(self) -> str:
        """The settings and scores as JSON, then any case the graph raised on."""
        body = self.model_dump_json(
            indent=2, include={"dataset_sha", "case_pattern", "settings", "metrics"}
        )
        errored = [f"  {r.case.id}  {r.error}" for r in self.results if r.error]
        return "\n".join([body, "", "errored:", *errored]) if errored else body
