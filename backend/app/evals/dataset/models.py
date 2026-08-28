"""Dataset values: the authored cases, the corpus they were authored against, and its drift."""

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from pydantic import Field, model_validator

from app.core.config import config
from app.core.models import FrozenModel
from app.evals.dataset.enums import EvalKind
from app.retrieval.models import ReferenceTarget

STAMP_LENGTH = 12
"""How much of a chunk's content hash a stamp carries. Short enough to read in a diff, long
enough that two chunks of a 20-case corpus cannot collide."""

SCORED_ONLY = {"cases": {"__all__": {"references": {"__all__": {"content_hashes"}}}}}
"""The stamps, excluded from the dataset hash: provenance, not anything a run scores."""


class CaseReference(ReferenceTarget):
    """A cited division, stamped with the chunks that covered it when the case was authored.

    Empty hashes mean unstamped, which is the different fact from stamped and unchanged.
    """

    content_hashes: tuple[str, ...] = ()


class CorpusStamp(FrozenModel):
    """The corpus the cases were authored against: its version, when it was read, and what
    each cited act hashed to, so a run can say whether the ground has moved since."""

    corpus_version: str | None = None
    stamped_at: date
    documents: dict[str, str] = Field(default_factory=dict)


class EvalCase(FrozenModel):
    """A question, what a right answer must say, and where in the corpus it comes from."""

    id: str
    kind: EvalKind
    question: str
    answer: str | None = None
    references: tuple[CaseReference, ...] = ()

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


class EmptyError(ValueError):
    """A dataset load that selected no cases."""


class EvalDataset(FrozenModel):
    """The golden dataset: every authored case, the corpus stamp they share, and the filter
    naming the subset a run scores."""

    cases: tuple[EvalCase, ...]
    corpus: CorpusStamp | None = None
    case_filter: str | None = None
    """Chosen per run rather than read from the file, so it is not written back out."""

    @classmethod
    def load(
        cls, path: Path = config.EVAL_DATASET_PATH, case_filter: str | None = None
    ) -> "EvalDataset":
        """Read and validate the JSON file at path."""
        dataset = cls.model_validate_json(path.read_bytes())
        if not dataset.cases:
            raise EmptyError("The dataset has no cases")

        dataset = dataset.model_copy(update={"case_filter": case_filter})
        if not dataset.selected_cases:
            raise EmptyError(f"No cases found matching filter: {case_filter}")
        return dataset

    def save(self, path: Path = config.EVAL_DATASET_PATH) -> None:
        """Write the corpus stamp and every case back out, one field per line and each
        reference on its own, so a re-stamp diffs as the lines that actually moved."""
        path.write_text(_format_dataset(self))

    @property
    def selected_cases(self) -> tuple[EvalCase, ...]:
        """The cases a run scores: those the filter matches, or every case without one."""
        if self.case_filter is None:
            return self.cases
        return tuple(case for case in self.cases if self.case_filter in case.id)

    @property
    def sha256(self) -> str:
        """Hash of what the cases assert, as canonical JSON — the whole dataset however a run
        filters it, so a filtered spot-check still names the file a full run scored.

        Stamps are left out: re-stamping records which text an answer was read against and
        changes nothing a run scores, so it must not break comparability with past runs.
        """
        scored = self.model_dump_json(include={"cases"}, exclude=SCORED_ONLY)
        return hashlib.sha256(scored.encode()).hexdigest()

    @property
    def unstamped_cases(self) -> tuple[str, ...]:
        """Cases citing a division no stamp was ever recorded for, so drift cannot be seen."""
        return tuple(
            case.id
            for case in self.cases
            if any(not reference.content_hashes for reference in case.references)
        )

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


class StaleReference(FrozenModel):
    """A case reference whose cited text changed since the case was stamped against it."""

    case_id: str
    target: ReferenceTarget
    stamped: tuple[str, ...]
    current: tuple[str, ...]


class ChangedDocument(FrozenModel):
    """A cited act whose bytes differ from what the dataset was stamped against."""

    celex: str
    stamped: str
    current: str | None
    """None where the act has left the standing corpus entirely."""


class DatasetDrift(FrozenModel):
    """How far the dataset has drifted from the corpus: what no longer resolves, what still
    resolves but has changed, which acts moved, and which cases were never stamped."""

    unresolved: tuple[UnresolvedReference, ...] = ()
    stale: tuple[StaleReference, ...] = ()
    changed_documents: tuple[ChangedDocument, ...] = ()
    unstamped: tuple[str, ...] = ()

    @property
    def stale_case_ids(self) -> tuple[str, ...]:
        """The stale cases named once each, however many of their references moved."""
        return tuple(dict.fromkeys(item.case_id for item in self.stale))


def _inline(payload: dict[str, object]) -> str:
    """A JSON object on one line, spaced as the hand-authored file spaces it."""
    return json.dumps(payload, separators=(", ", ": "), ensure_ascii=False)


def _format_case(case: EvalCase) -> str:
    """One case: its scalars a line each, then every reference on a line of its own."""
    fields = case.model_dump(mode="json", exclude={"references"}, exclude_defaults=True)
    lines = [f"      {json.dumps(name)}: {_inline(value)}" for name, value in fields.items()]
    if case.references:
        references = ",\n".join(
            f"        {_inline(reference.model_dump(mode='json', exclude_defaults=True))}"
            for reference in case.references
        )
        lines.append(f'      "references": [\n{references}\n      ]')
    body = ",\n".join(lines)
    return f"    {{\n{body}\n    }}"


def _format_dataset(dataset: EvalDataset) -> str:
    """The whole file: the corpus stamp, then the cases in authored order."""
    corpus = json.dumps(
        dataset.corpus.model_dump(mode="json") if dataset.corpus else None, indent=2
    )
    corpus = "\n".join(
        line if index == 0 else f"  {line}" for index, line in enumerate(corpus.splitlines())
    )
    cases = ",\n".join(_format_case(case) for case in dataset.cases)
    return f'{{\n  "corpus": {corpus},\n  "cases": [\n{cases}\n  ]\n}}\n'
