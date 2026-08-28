"""Dataset CLI: the `evals check` and `evals stamp` subcommands."""

import asyncio
from typing import Any

from app.core.db.session import get_session
from app.evals.dataset.check import check_against_corpus, find_moved_corpus, format_drift
from app.evals.dataset.enums import DriftKind
from app.evals.dataset.models import EvalDataset
from app.evals.dataset.stamp import save_dataset, stamp_dataset


def register_dataset_commands(commands: Any) -> None:
    """The read-only check, and the stamp that records a human has re-read the cited text."""
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
        "new text. An unfiltered stamp also rewrites the corpus stamp; --case leaves it alone.",
    )
    stamp.add_argument("--case", help="only cases whose id contains this")


def run_check() -> int:
    """Report every way the dataset has drifted; only an unresolved reference fails."""
    dataset = EvalDataset.load()
    drifted, current = asyncio.run(check_against_corpus(dataset))
    lines = format_drift(drifted, find_moved_corpus(dataset, current))
    print("\n".join(lines) if lines else "every case reference resolves and is stamped")
    return 1 if any(item.kind is DriftKind.UNRESOLVED for item in drifted) else 0


async def _stamp_against_corpus(dataset: EvalDataset) -> EvalDataset:
    """The dataset restamped in one session against the corpus as it stands."""
    async with get_session(auto_commit=False) as session:
        return await stamp_dataset(session, dataset)


def run_stamp(case_filter: str | None) -> int:
    """Restamp the selected cases, write the dataset back, and name whose hashes moved."""
    before = EvalDataset.load(case_filter=case_filter)
    after = asyncio.run(_stamp_against_corpus(before))
    save_dataset(after)

    changed = [
        case.id
        for case, was in zip(after.cases, before.cases, strict=True)
        if case.references != was.references
    ]
    print(f"stamped {len(after.selected_cases)} cases")
    print(f"hashes changed: {', '.join(changed)}" if changed else "no hashes changed")
    return 0
