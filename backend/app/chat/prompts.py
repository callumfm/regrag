"""System prompt, the context budget, and numbered-context formatting for the chat graph."""

from collections.abc import Sequence
from itertools import groupby

from app.retrieval.expand import SectionKey
from app.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You are RegRag, an assistant answering questions about EU maritime regulation. "
    "Answer using only the numbered context blocks provided. Cite every claim inline "
    "with the marker of the block it comes from, like [1] or [2][3]. If the context "
    "does not answer the question, say so plainly instead of guessing."
)


def fit_context(sources: Sequence[RetrievedChunk], max_chars: int) -> tuple[RetrievedChunk, ...]:
    """The leading whole sections of the sources that fit the budget, the first always.

    Sources arrive section by section in the rank each was found at, so what is dropped
    is the least relevant; a section is kept whole or not at all, since widening to the
    whole of it was the point.
    """
    kept: list[RetrievedChunk] = []
    used = 0
    for _, group in groupby(sources, key=SectionKey.from_chunk):
        section = list(group)
        size = sum(len(chunk.text) for chunk in section)
        if kept and used + size > max_chars:
            break
        kept.extend(section)
        used += size
    return tuple(kept)


def format_context(sources: Sequence[RetrievedChunk]) -> str:
    """The retrieved chunks as numbered blocks the citation markers refer to."""
    blocks = [
        f"[{marker}] ({source.celex}, {source.citation})\n{source.text}"
        for marker, source in enumerate(sources, start=1)
    ]
    return "\n\n".join(blocks)


def build_user_message(question: str, sources: Sequence[RetrievedChunk]) -> str:
    """The full user turn: context blocks first, then the question."""
    return f"Context:\n\n{format_context(sources)}\n\nQuestion: {question}"
