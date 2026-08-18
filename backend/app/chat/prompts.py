"""The chat graph's fixed wording: system prompt, refusal, numbered-context formatting."""

from collections.abc import Sequence

from app.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You are RegRag, an assistant answering questions about EU maritime regulation. "
    "Answer using only the numbered context blocks provided. Cite every claim inline "
    "with the marker of the block it comes from, like [1] or [2][3]. If the context "
    "does not answer the question, say so plainly instead of guessing."
)

REFUSAL_ANSWER = (
    "The corpus doesn't cover this. RegRag answers questions about the EU maritime "
    "regulation it has ingested; try asking about that."
)


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
