"""Eval values: the golden dataset and the cases it holds."""

import hashlib
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter, computed_field, model_validator

from app.core.clock import utc_now
from app.core.config import config
from app.core.models import FrozenModel
from app.evals.enums import EvalKind
from app.evals.metrics import (
    is_gate_refusal,
    score_citation_validity,
    score_reference_citation_rate,
    score_reference_recall,
)
from app.retrieval.models import ReferenceTarget, RetrievedChunk, SearchResult


def _mean(values: Sequence[float | bool]) -> float | None:
    """The mean, or None when there is nothing to average — unmeasured, not zero."""
    return sum(values) / len(values) if values else None


def _score(value: float | None) -> str:
    """A score to two places, or a dash holding its column when it was not measured."""
    return f"{value:>4.2f}" if value is not None else f"{'-':>4}"


class EvalCase(FrozenModel):
    """A question, what a right answer must say, and where in the corpus it comes from."""

    id: str
    kind: EvalKind
    question: str
    answer: str | None = None
    references: tuple[ReferenceTarget, ...] = ()

    @model_validator(mode="after")
    def _kind_matches_fields(self) -> "EvalCase":
        """An in-corpus case is scored against its answer and references; an out-of-corpus case
        is scored on refusal alone, so carrying either would be a mislabelled case."""
        has_evidence = bool(self.references) and self.answer is not None
        if self.kind is EvalKind.IN_CORPUS and not has_evidence:
            raise ValueError(f"{self.id}: an in_corpus case needs an answer and references")
        if self.kind is EvalKind.OUT_OF_CORPUS and (self.references or self.answer):
            raise ValueError(f"{self.id}: an out_of_corpus case has neither answer nor references")
        return self


_cases = TypeAdapter(tuple[EvalCase, ...])


class EvalDataset(FrozenModel):
    """The golden dataset: every authored case."""

    cases: tuple[EvalCase, ...]

    @classmethod
    def load(cls, path: Path = config.EVAL_DATASET_PATH) -> "EvalDataset":
        """Read and validate the JSON file at path."""
        return cls(cases=_cases.validate_json(path.read_bytes()))

    @property
    def sha256(self) -> str:
        """Hash of the cases as canonical JSON, so a run records exactly which dataset it scored."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "EvalDataset":
        """A case id is how a result names its case, so two cases cannot share one."""
        counts = Counter(case.id for case in self.cases)
        duplicates = sorted(id for id, seen in counts.items() if seen > 1)
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        return self


class UnresolvedReference(FrozenModel):
    """A case reference no stored chunk answers to."""

    case_id: str
    target: ReferenceTarget


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

    hits: the raw search results, before the gate and before expansion.
    error: what the graph raised; such a case is counted but not scored.
    """

    case: EvalCase
    hits: tuple[SearchResult, ...] = ()
    sources: tuple[RetrievedChunk, ...] = ()
    answer: str = ""
    retrieve_ms: int = 0
    total_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    @computed_field
    @property
    def synthesize_ms(self) -> int:
        """What the model call cost, once retrieval had already run."""
        return self.total_ms - self.retrieve_ms

    @property
    def raw_recall(self) -> float:
        """Share of gold references search itself found, before expansion widened anything."""
        return score_reference_recall(self.case.references, self.hits)

    @property
    def expanded_recall(self) -> float:
        """Share of gold references that reached the prompt."""
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
        """Whether the score gate refused before any model call."""
        return is_gate_refusal(self.answer)

    @property
    def refused_a_covered_case(self) -> bool:
        """An in-corpus case refused though search had found a reference: the gate too
        tight, rather than a corpus that does not cover the question."""
        return self.case.kind is EvalKind.IN_CORPUS and self.gate_refused and self.raw_recall > 0

    def line(self) -> str:
        """One case on one line: both recalls, both citation scores, outcome, cost, with a
        dash where there was nothing to score."""
        recalled = self.case.references and not self.error
        raw = _score(self.raw_recall if recalled else None)
        expanded = _score(self.expanded_recall if recalled else None)
        cited = _score(self.reference_citation_rate if recalled else None)
        valid = _score(self.citation_validity if not self.error else None)
        outcome = "error" if self.error else ("gate-refused" if self.gate_refused else "answered")
        tokens = f"{self.input_tokens or 0}/{self.output_tokens or 0}"
        line = (
            f"{self.case.id:<44} raw {raw}  exp {expanded}  cite {cited}  ok {valid}  "
            f"{outcome:<12} {self.total_ms / 1000:>5.1f}s {tokens:>12}"
        )
        return f"{line}  {self.error}" if self.error else line


class RunMetrics(FrozenModel):
    """What a run measured over the cases that completed; a rate is None when the run held
    no case it applies to, which is unmeasured rather than zero."""

    cases: int
    in_corpus: int
    out_of_corpus: int
    errors: int
    raw_hit_rate: float | None
    raw_recall: float | None
    expanded_hit_rate: float | None
    expanded_recall: float | None
    reference_citation_rate: float | None
    citation_validity: float | None
    gate_refusal_rate: float | None
    false_refusals: int
    gate_refused_a_found_reference: int
    input_tokens: int
    output_tokens: int
    mean_retrieve_ms: int
    mean_total_ms: int

    @classmethod
    def from_cases(cls, results: Sequence[CaseResult]) -> "RunMetrics":
        """Aggregate the completed cases; an errored case counts toward `errors` only."""
        scored = [result for result in results if result.error is None]
        in_corpus = [r for r in scored if r.case.kind is EvalKind.IN_CORPUS]
        out_of_corpus = [r for r in scored if r.case.kind is EvalKind.OUT_OF_CORPUS]
        cite_rates = [
            r.reference_citation_rate for r in in_corpus if r.reference_citation_rate is not None
        ]
        validities = [r.citation_validity for r in scored if r.citation_validity is not None]
        return cls(
            cases=len(results),
            in_corpus=len(in_corpus),
            out_of_corpus=len(out_of_corpus),
            errors=len(results) - len(scored),
            raw_hit_rate=_mean([r.raw_recall > 0 for r in in_corpus]),
            raw_recall=_mean([r.raw_recall for r in in_corpus]),
            expanded_hit_rate=_mean([r.expanded_recall > 0 for r in in_corpus]),
            expanded_recall=_mean([r.expanded_recall for r in in_corpus]),
            reference_citation_rate=_mean(cite_rates),
            citation_validity=_mean(validities),
            gate_refusal_rate=_mean([r.gate_refused for r in out_of_corpus]),
            false_refusals=sum(r.gate_refused for r in in_corpus),
            gate_refused_a_found_reference=sum(r.refused_a_covered_case for r in in_corpus),
            input_tokens=sum(r.input_tokens or 0 for r in scored),
            output_tokens=sum(r.output_tokens or 0 for r in scored),
            mean_retrieve_ms=int(_mean([r.retrieve_ms for r in scored]) or 0),
            mean_total_ms=int(_mean([r.total_ms for r in scored]) or 0),
        )


class RunResult(FrozenModel):
    """One eval run: when it ran, which dataset and settings it scored, and every case.

    dataset_sha hashes the whole dataset; case_pattern names the subset actually scored.
    """

    started_at: datetime
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
        started_at: datetime | None = None,
        case_pattern: str | None = None,
    ) -> "RunResult":
        """A finished run, with its cases aggregated and the settings they ran under."""
        return cls(
            started_at=started_at or utc_now(),
            dataset_sha=dataset_sha,
            case_pattern=case_pattern,
            settings=RunSettings.from_config(),
            metrics=RunMetrics.from_cases(results),
            results=tuple(results),
        )

    def table(self) -> str:
        """The per-case rows, then the run's totals, the expansion row only when it ran."""
        metrics = self.metrics
        scope = f"  cases matching {self.case_pattern!r}" if self.case_pattern else ""
        rows = [result.line() for result in self.results]
        summary = [
            "",
            f"cases {metrics.cases}  in-corpus {metrics.in_corpus}  "
            f"out-of-corpus {metrics.out_of_corpus}  errors {metrics.errors}{scope}",
            f"hit-rate@{self.settings.chat_sources}      {_score(metrics.raw_hit_rate)}"
            f"      recall {_score(metrics.raw_recall)}   (raw search hits)",
        ]
        if self.settings.expand_sections:
            summary.append(
                f"hit-rate@<={self.settings.chat_context_chunks}   "
                f"{_score(metrics.expanded_hit_rate)}"
                f"      recall {_score(metrics.expanded_recall)}   (after section expansion)"
            )
        summary += [
            f"cited references   {_score(metrics.reference_citation_rate)}"
            f"      markers in context {_score(metrics.citation_validity)}",
            "  (an answer citing a further relevant article is not penalised; whether a"
            " citation supports its claim is the judge's to score)",
            f"gate refusal rate  {_score(metrics.gate_refusal_rate)}"
            f"      false refusals {metrics.false_refusals}"
            f"   over a found reference {metrics.gate_refused_a_found_reference}",
            "  (the cheap pre-model gate only; a model-worded decline is the judge's to score)",
            f"tokens {metrics.input_tokens}/{metrics.output_tokens}  "
            f"mean retrieve {metrics.mean_retrieve_ms}ms  mean total {metrics.mean_total_ms}ms",
        ]
        return "\n".join(rows + summary)
