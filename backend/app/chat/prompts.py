"""System prompt and numbered-context formatting for the chat graph."""

from app.chat.models import numbered
from app.retrieval.models import SearchResult

SYSTEM_PROMPT = (
    "You are RegRag, an assistant answering questions about EU maritime regulation. "
    "Answer using only the numbered context blocks provided. Cite every claim inline "
    "with the marker of the block it comes from, like [1] or [2][3]. If the context "
    "does not answer the question, say so plainly instead of guessing."
)


def format_context(sources: tuple[SearchResult, ...]) -> str:
    """The retrieved chunks as numbered blocks the citation markers refer to."""
    blocks = [
        f"[{marker}] ({source.celex}, {source.citation})\n{source.text}"
        for marker, source in numbered(sources)
    ]
    return "\n\n".join(blocks)


def build_user_message(question: str, sources: tuple[SearchResult, ...]) -> str:
    """The full user turn: context blocks first, then the question."""
    return f"Context:\n\n{format_context(sources)}\n\nQuestion: {question}"
