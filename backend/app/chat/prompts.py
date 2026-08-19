"""The chat graph's fixed wording: system prompt, refusal, numbered-context formatting."""

from collections.abc import Sequence

from app.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You are RegRag, an assistant answering questions about EU maritime regulation. "
    "Answer using only the numbered context blocks provided. Cite every claim inline "
    "with the marker of the block it comes from, like [1] or [2][3], placed after the "
    "punctuation that ends the claim (e.g. 'must be reported.[1]'), never before it. "
    "If the context "
    "does not answer the question, say so plainly instead of guessing. "
    "Start directly with the answer: no title, no restating the question, and no "
    "preamble such as 'Based on the context provided'. When several acts give the same "
    "answer, give it once and name the acts it holds for, then note only where they "
    "differ; do not repeat near-identical lists per act. Refer to an act by the number "
    "the context gives it; never invent a name or title for it."
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
