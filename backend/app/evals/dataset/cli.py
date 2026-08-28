"""The dataset subcommands, registered into the evals parser."""

from typing import Any


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
        "new text. An unfiltered stamp also rewrites the corpus stamp; --case leaves it alone.",
    )
    stamp.add_argument("--case", help="only cases whose id contains this")
