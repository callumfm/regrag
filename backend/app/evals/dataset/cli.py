"""Dataset CLI: the `evals check` and `evals stamp` subcommands."""

import asyncio
from typing import Any

from app.core.db.session import get_session
from app.evals.dataset.models import DatasetDrift, EvalDataset
from app.evals.dataset.service import inspect_dataset, stamp_dataset


def register_dataset_commands(commands: Any) -> None:
    """The read-only check and the write that records a re-review, kept apart on purpose:
    stamping asserts a human has read the cited text, which no check can do on their behalf."""
    commands.add_parser(
        "check",
        help="confirm every case reference resolves and still says what it was authored against",
        description="Report hard drift (a reference no stored chunk answers to), soft drift "
        "(cited text that changed since the case was stamped), and any case never stamped. "
        "Read-only: repairing a stale case means rewriting its answer, then `evals stamp`.",
    )
    stamp = commands.add_parser(
        "stamp",
        help="record what the cited text says now, asserting it has been read",
        description="Write the current chunk hashes into the selected cases. Run it on a newly "
        "authored case, or on a stale one whose answer you have just re-reviewed against the "
        "new text. An unfiltered stamp also rewrites the corpus block; --case leaves it alone.",
    )
    stamp.add_argument("--case", help="only cases whose id contains this")


def format_drift(drift: DatasetDrift) -> list[str]:
    """Every drift signal as its own block, most actionable first: what broke, what moved
    underneath a case, which acts changed, and what was never stamped at all."""
    lines = [
        f"{item.case_id}: no stored chunk for {item.target.celex} {item.target.citation}"
        for item in drift.unresolved
    ]
    if drift.stale:
        width = max(len(item.case_id) for item in drift.stale)
        lines.append("stale (cited text changed since authoring):")
        lines += [
            f"  {item.case_id:<{width}}  {item.target.celex} {item.target.citation}"
            for item in drift.stale
        ]
    lines += [
        f"{item.celex} changed since stamping "
        f"({item.stamped[:12]} -> {item.current[:12] if item.current else 'gone'})"
        for item in drift.changed_documents
    ]
    if drift.unstamped:
        lines.append(f"{len(drift.unstamped)} cases unstamped: {', '.join(drift.unstamped)}")
    return lines


async def _inspect() -> DatasetDrift:
    async with get_session(auto_commit=False) as session:
        return await inspect_dataset(session, EvalDataset.load())


def check_dataset() -> int:
    """Report every way the dataset has drifted from the corpus. Only an unresolved reference
    fails: a stale case needs a human re-review, not a red build."""
    drift = asyncio.run(_inspect())
    lines = format_drift(drift)
    print("\n".join(lines) if lines else "every case reference resolves and is stamped")
    return 1 if drift.unresolved else 0


async def _stamp(case_filter: str | None) -> tuple[EvalDataset, EvalDataset]:
    async with get_session(auto_commit=False) as session:
        before = EvalDataset.load(case_filter=case_filter)
        return before, await stamp_dataset(session, before)


def stamp_cases(case_filter: str | None) -> int:
    """Restamp the selected cases and write the dataset back, naming what moved."""
    before, after = asyncio.run(_stamp(case_filter))
    after.save()

    changed = [
        case.id
        for case, was in zip(after.cases, before.cases, strict=True)
        if case.references != was.references
    ]
    print(f"stamped {len(after.selected_cases)} cases")
    print(f"hashes changed: {', '.join(changed)}" if changed else "no hashes changed")
    return 0
