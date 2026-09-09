"""The assess loop's tool surface: what the model may call, and how each call runs."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import NamedTuple

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ChatStepStatus, ToolStep
from app.chat.models import ChatStepResult, ToolCall
from app.core.config import config
from app.core.db.session import get_session
from app.core.llm import LLMError
from app.core.models import FrozenModel
from app.retrieval.follow import follow_reference
from app.retrieval.models import ReferenceTarget, RetrievedChunk, SearchFilters, SearchRequest
from app.retrieval.search import search
from app.retrieval.thresholds import meets_thresholds

logger = logging.getLogger(__name__)


class SearchArgs(FrozenModel):
    """A fresh corpus search: for what is named in the context but not cited."""

    query: str
    celex: str | None = None


class FollowReferenceArgs(FrozenModel):
    """A cited division to fetch outright, addressed as the citation addresses it."""

    celex: str
    article: str | None = None
    paragraph: str | None = None
    annex: str | None = None


async def run_search(session: AsyncSession, args: SearchArgs) -> tuple[RetrievedChunk, ...]:
    """A call's hits, or nothing when they miss the bar retrieval holds its own hits to:
    what the gate refuses to answer from, the loop may not add to the context either."""
    request = SearchRequest(
        query=args.query,
        filters=SearchFilters(celex=args.celex),
        limit=config.ASSESS_SEARCH_LIMIT,
    )
    hits = await search(session, request)
    return hits if meets_thresholds(hits) else ()


async def run_follow_reference(
    session: AsyncSession, args: FollowReferenceArgs
) -> tuple[RetrievedChunk, ...]:
    """The division's text from the top, capped: a long article or annex would otherwise
    spend the round's whole budget on one call. No score to gate on — the context cited it."""
    target = ReferenceTarget(
        celex=args.celex, article=args.article, paragraph=args.paragraph, annex=args.annex
    )
    chunks = await follow_reference(session, target)
    return chunks[: config.ASSESS_FOLLOW_LIMIT]


class RefuseArgs(FrozenModel):
    """Assess's word that nothing in the context bears on the question and no fetch would:
    the one call that ends the question in the fixed refusal rather than an answer."""

    reason: str


def tool_definition(name: str, description: str, args_model: type[FrozenModel]) -> dict:
    """A tool as bind_tools wants it: an openai function-tool dictionary."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": args_model.model_json_schema(),
        },
    }


class ToolSpec(NamedTuple):
    """One tool the model may call to grow the context: how it is named and described to
    the model, the arguments it takes, what runs it, and the step a call to it records."""

    name: str
    step: ToolStep
    args_model: type[FrozenModel]
    run: Callable[..., Awaitable[tuple[RetrievedChunk, ...]]]
    description: str

    def definition(self) -> dict:
        return tool_definition(self.name, self.description, self.args_model)


TOOL_SURFACE = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "search",
            ToolStep.SEARCH,
            SearchArgs,
            run_search,
            "Search the corpus for text matching a query, optionally within one act (celex). "
            "Use for concepts the context names without citing, or question parts with no "
            "context at all.",
        ),
        ToolSpec(
            "follow_reference",
            ToolStep.FOLLOW_REFERENCE,
            FollowReferenceArgs,
            run_follow_reference,
            "Fetch the full text of one cited division: an article (optionally one paragraph) "
            "or an annex of an act (celex). Use the addresses on the context's cites lines.",
        ),
    )
}

FETCH_TOOL_DEFINITIONS = [spec.definition() for spec in TOOL_SURFACE.values()]
"""The tools that grow the context, as the model is shown them: built once, since they
depend on nothing at call time."""

REFUSE_TOOL_NAME = "refuse"
REFUSE_TOOL_DEFINITION = tool_definition(
    REFUSE_TOOL_NAME,
    "Declare that nothing in the context bears on the question and no search or fetch of "
    "this corpus could change that. Call it alone, never beside a search or fetch.",
    RefuseArgs,
)


def tool_definitions() -> list[dict]:
    """The surface as assess is shown it: the fetch tools, and refuse while it is allowed."""
    if config.ASSESS_MAY_REFUSE:
        return [*FETCH_TOOL_DEFINITIONS, REFUSE_TOOL_DEFINITION]
    return list(FETCH_TOOL_DEFINITIONS)


def is_refusal(call: ToolCall) -> bool:
    """Whether the call is assess's refusal rather than a fetch."""
    return call.name == REFUSE_TOOL_NAME


def already_in_context(call: ToolCall, sources: Sequence[RetrievedChunk]) -> bool:
    """Whether the call would only fetch a paragraph the context already shows in full, so
    running it could add nothing. A whole article or annex is never known to be shown in
    full: its chapeau's parts say nothing about what sits under it. A call the surface cannot
    read is left to run_tool_call to reject."""
    if call.name != "follow_reference":
        return False
    try:
        target = ReferenceTarget.model_validate(call.args)
    except ValidationError:
        return False
    if target.paragraph is None:
        return False
    citation = target.citation.lower()
    shown = [s for s in sources if s.celex == target.celex and s.citation.lower() == citation]
    return bool(shown) and len({s.part for s in shown}) == shown[0].parts


def describe_call(call: ToolCall) -> str | None:
    """What a call was for, as the trail shows it: the arguments it was given, in the order
    the model gave them — a query, or a citation's address — and nothing when it gave none."""
    given = [str(value) for value in call.args.values() if value]
    return " · ".join(given) or None


def tool_step(name: str) -> ToolStep:
    """The step a call to this tool records; a tool the surface does not have records that."""
    spec = TOOL_SURFACE.get(name)
    return spec.step if spec else ToolStep.UNKNOWN


def build_call_step(
    call: ToolCall, *, ms: int = 0, status: ChatStepStatus = ChatStepStatus.COMPLETED
) -> ChatStepResult:
    """The step a call records, as announced when it starts and as settled once it has run:
    built in one place so the two frames the client pairs up cannot disagree."""
    return ChatStepResult(
        step=tool_step(call.name), ms=ms, status=status, subject=describe_call(call)
    )


async def run_tool_call(call: ToolCall) -> tuple[RetrievedChunk, ...]:
    """One call's chunks; an unknown tool, an invalid target or a failing call yields
    nothing, never an error — the loop is best-effort and a bad call adds nothing.

    The session is the call's own, so a database error rolls back only the call that hit it;
    shared, the rollback it leaves owing would fail every later call in the round.
    """
    spec = TOOL_SURFACE.get(call.name)
    if spec is None:
        logger.warning("assess called unknown tool %s", call.name)
        return ()
    try:
        args = spec.args_model.model_validate(call.args)
        async with get_session(auto_commit=False) as session:
            return await spec.run(session, args)
    except ValidationError:
        logger.warning("assess called %s with invalid arguments", call.name)
        return ()
    except (LLMError, SQLAlchemyError) as exc:
        logger.warning("assess call to %s failed: %s", call.name, exc)
        return ()
