"""The chat graph's fixed wording: system prompt, refusal, numbered-context formatting."""

import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from app.chat.models import ChatTurn
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


def build_user_message(question: str, sources: Sequence[RetrievedChunk]) -> str:
    """The full user turn: context blocks first, then the question."""
    return f"Context:\n\n{format_context(sources)}\n\nQuestion: {question}"


ASSESS_SYSTEM_PROMPT = (
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

ASSESS_REFUSAL_INSTRUCTION = (
    " If no block bears on the question and no search or fetch of this corpus of EU "
    "maritime regulation could — it asks about another regime, about a named company, "
    "ship or event, for a statistic or a figure no provision states, or about a topic "
    "outside the corpus — call refuse, alone, saying why. Blocks on the "
    "subject the question touches that do not answer it are not a part answer. Never call "
    "it on a question the context answers in part, or one a search or fetch might yet "
    "answer."
)


def build_assess_system_prompt(*, may_refuse: bool) -> str:
    """The assess system prompt, telling the model when to refuse only when it is offered
    the tool to do it with."""
    return ASSESS_SYSTEM_PROMPT + (ASSESS_REFUSAL_INSTRUCTION if may_refuse else "")


def build_assess_message(question: str, sources: Sequence[RetrievedChunk]) -> str:
    """The full assess turn: the same numbered blocks synthesize will cite, each with the
    addresses it cites, then the question."""
    return f"Context:\n\n{format_context(sources, cites=True)}\n\nQuestion: {question}"


def thread_messages(history: Sequence[ChatTurn]) -> list[BaseMessage]:
    """The thread's earlier turns as the message pairs a model reads them as, oldest first,
    to go between the system prompt and this turn's user message."""
    messages: list[BaseMessage] = []
    for turn in history:
        messages.append(HumanMessage(turn.question))
        messages.append(AIMessage(turn.answer))
    return messages


REWRITE_SYSTEM_PROMPT = (
    "You restate the latest question of a conversation about EU maritime regulation so "
    "that it can be searched on its own, without the conversation. You are shown the "
    "earlier turns, then the question. Replace pronouns and shorthand — 'it', 'that "
    "regulation', 'the penalties' — with what the earlier turns show they refer to, and "
    "carry over the act or scheme the conversation is about when the question leaves it "
    "unsaid. Never answer, and never add anything the question did not ask. A question "
    "that already stands on its own is returned unchanged."
)


def build_rewrite_message(question: str, history: Sequence[ChatTurn]) -> str:
    """The full rewrite turn: the thread as a transcript, then the question to restate."""
    transcript = "\n\n".join(f"Q: {turn.question}\nA: {turn.answer}" for turn in history)
    return f"Conversation so far:\n\n{transcript}\n\nLatest question: {question}"


DECOMPOSE_SYSTEM_PROMPT = (
    "You split a question about EU maritime regulation into the separate searches it "
    "needs, one per distinct thing it asks. A question asking one thing, however long, "
    "is one query: return it unchanged. Split only when the parts would be answered by "
    "different provisions; never split a single obligation into its conditions, and "
    "never rephrase, narrow or expand what was asked. Each query must stand alone, "
    "naming the act or scheme the question names, so that searching it without the "
    "others finds the right provision."
)
