"""The wording more than one node shares: the thread note, and the numbered-context
formatting a citation marker refers to."""

import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from app.chat.models import ChatTurn
from app.retrieval.models import ReferenceTarget, RetrievedChunk

THREAD_NOTE = (
    " Earlier turns of the conversation come before the context; read them only to "
    "understand what the question refers to. They are not context to answer from: cite "
    "only this turn's numbered blocks."
)


def system_prompt(base: str, history: Sequence[ChatTurn]) -> str:
    """The system prompt as one turn sends it: the base alone on a first question, and
    with the thread note on a follow-up, so a first question's prompt is unchanged."""
    return f"{base}{THREAD_NOTE}" if history else base


MARKER = re.compile(r"\[(\d+)\]")
"""A citation marker as the system prompt asks for it, like [1] or the [2][3] of a pair."""


def strip_markers(text: str) -> str:
    """The text without its citation markers: an earlier answer carried into a later turn
    numbered blocks that turn will not have."""
    return MARKER.sub("", text)


def _reference_addresses(source: RetrievedChunk) -> list[str]:
    """Each followable address once, as 'celex division': a reference naming no division is
    skipped, on the same rule follow_reference's target enforces, and several points of one
    article are the one address."""
    addresses = []
    for reference in source.references:
        try:
            target = ReferenceTarget.from_reference(reference, citing=source.celex)
        except ValidationError:
            continue
        addresses.append(f"{target.celex} {target.citation}")
    return list(dict.fromkeys(addresses))


def format_context_block(marker: int, source: RetrievedChunk) -> str:
    """One chunk as the numbered block a citation marker refers to."""
    return f"[{marker}] ({source.celex}, {source.citation})\n{source.text}"


def format_context(sources: Sequence[RetrievedChunk], *, cites: bool = False) -> str:
    """The retrieved chunks as numbered blocks the citation markers refer to, each block
    followed by the addresses it cites when the caller asks for them."""
    blocks = []
    for marker, source in enumerate(sources, start=1):
        block = format_context_block(marker, source)
        if cites and (addresses := _reference_addresses(source)):
            block += f"\ncites: {', '.join(addresses)}"
        blocks.append(block)
    return "\n\n".join(blocks)


def thread_messages(history: Sequence[ChatTurn]) -> list[BaseMessage]:
    """The thread's earlier turns as the message pairs a model reads them as, oldest first,
    to go between the system prompt and this turn's user message."""
    messages: list[BaseMessage] = []
    for turn in history:
        messages.append(HumanMessage(turn.question))
        messages.append(AIMessage(turn.answer))
    return messages
