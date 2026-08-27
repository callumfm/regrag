"""The chat graph's fixed wording: system prompt, refusal, numbered-context formatting."""

from collections.abc import Sequence

from app.retrieval.models import ReferenceTarget, RetrievedChunk

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


GATHER_SYSTEM_PROMPT = (
    "You decide what RegRag, an assistant answering questions about EU maritime "
    "regulation, still needs to read before answering. You are shown a question and "
    "the numbered context blocks retrieved so far; each block may list what it cites. "
    "If the context already answers the whole question, call no tools. Otherwise call "
    "what fills the gap: follow_reference fetches the exact text a block cites — "
    "prefer it whenever a block leans on a provision named in its cites line, passing "
    "that line's document number and division; search runs a fresh corpus search — "
    "use it when a needed concept is named without a citation, or a part of the "
    "question has no context at all, narrowing with celex when the act is known. "
    "Never re-fetch what the context already shows. You never answer the question "
    "yourself: your output is tool calls, or nothing when the context suffices."
)


def _reference_addresses(source: RetrievedChunk) -> list[str]:
    """Each followable reference as 'celex division'; one naming no division is skipped."""
    addresses = []
    for reference in source.references:
        if reference.article is None and reference.annex is None:
            continue
        target = ReferenceTarget.from_reference(reference, citing=source.celex)
        addresses.append(f"{target.celex} {target.citation}")
    return addresses


def format_gather_context(sources: Sequence[RetrievedChunk]) -> str:
    """The numbered blocks as gather sees them: each with the addresses it cites."""
    blocks = []
    for marker, source in enumerate(sources, start=1):
        block = f"[{marker}] ({source.celex}, {source.citation})\n{source.text}"
        if addresses := _reference_addresses(source):
            block += f"\ncites: {', '.join(addresses)}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_gather_message(question: str, sources: Sequence[RetrievedChunk]) -> str:
    """The full gather turn: context blocks with their citations, then the question."""
    return f"Context:\n\n{format_gather_context(sources)}\n\nQuestion: {question}"
