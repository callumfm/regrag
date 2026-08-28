"""`evals stamp`: record what the cited text says now, asserting that it has been read."""

import asyncio
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import utc_today
from app.core.config import config
from app.core.db.session import get_session
from app.evals.dataset.check import current_stamp
from app.evals.dataset.models import CorpusStamp, EvalCase, EvalDataset
from app.ingestion.service import get_latest_corpus_version


async def _stamp_case(session: AsyncSession, case: EvalCase) -> EvalCase:
    """The case with every reference carrying what its division hashes to now."""
    references = [
        reference.model_copy(update={"content_hashes": await current_stamp(session, reference)})
        for reference in case.references
    ]
    return case.model_copy(update={"references": tuple(references)})


async def stamp_dataset(session: AsyncSession, dataset: EvalDataset) -> EvalDataset:
    """The dataset restamped against the corpus as it stands.

    Only the selected cases are stamped, so repairing one case cannot silently clear the
    staleness of the others. The corpus stamp covers the whole dataset, so it is rewritten
    only by an unfiltered stamp, which is the one that can honestly claim to.
    """
    selected = {case.id for case in dataset.selected_cases}
    cases = [
        await _stamp_case(session, case) if case.id in selected else case for case in dataset.cases
    ]
    corpus = (
        dataset.corpus
        if dataset.case_filter
        else CorpusStamp(
            corpus_version=await get_latest_corpus_version(session), stamped_at=utc_today()
        )
    )
    return dataset.model_copy(update={"cases": tuple(cases), "corpus": corpus})


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
    return "    {\n" + ",\n".join(lines) + "\n    }"


def format_dataset(dataset: EvalDataset) -> str:
    """The whole file: the corpus stamp, then the cases in authored order, one field per line
    and each reference on its own, so a re-stamp diffs as the lines that actually moved."""
    stamp = json.dumps(dataset.corpus.model_dump(mode="json") if dataset.corpus else None, indent=2)
    stamp = "\n  ".join(stamp.splitlines())
    cases = ",\n".join(_format_case(case) for case in dataset.cases)
    return f'{{\n  "corpus": {stamp},\n  "cases": [\n{cases}\n  ]\n}}\n'


def save_dataset(dataset: EvalDataset, path: Path = config.EVAL_DATASET_PATH) -> None:
    """Write the dataset back out. Only stamping writes the file, so only stamping formats it."""
    path.write_text(format_dataset(dataset))


async def _stamp(case_filter: str | None) -> tuple[EvalDataset, EvalDataset]:
    async with get_session(auto_commit=False) as session:
        before = EvalDataset.load(case_filter=case_filter)
        return before, await stamp_dataset(session, before)


def stamp_cases(case_filter: str | None) -> int:
    """Restamp the selected cases and write the dataset back, naming what moved."""
    before, after = asyncio.run(_stamp(case_filter))
    save_dataset(after)

    changed = [
        case.id
        for case, was in zip(after.cases, before.cases, strict=True)
        if case.references != was.references
    ]
    print(f"stamped {len(after.selected_cases)} cases")
    print(f"hashes changed: {', '.join(changed)}" if changed else "no hashes changed")
    return 0
