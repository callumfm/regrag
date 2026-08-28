"""The assess loop's tool surface: what the model may call, and how each call runs."""

import logging
from collections.abc import Awaitable, Callable
from typing import NamedTuple

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ToolStep
from app.chat.models import ToolCall
from app.core.config import config
from app.core.llm import LLMError
from app.core.models import FrozenModel
from app.retrieval.follow import follow_reference
from app.retrieval.models import ReferenceTarget, RetrievedChunk, SearchFilters, SearchRequest
from app.retrieval.search import search

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
    request = SearchRequest(
        query=args.query,
        filters=SearchFilters(celex=args.celex),
        limit=config.ASSESS_SEARCH_LIMIT,
    )
    return await search(session, request)


async def run_follow_reference(
    session: AsyncSession, args: FollowReferenceArgs
) -> tuple[RetrievedChunk, ...]:
    target = ReferenceTarget(
        celex=args.celex, article=args.article, paragraph=args.paragraph, annex=args.annex
    )
    return await follow_reference(session, target)


class ToolSpec(NamedTuple):
    """One tool the model may call: how it is named and described to the model, the
    arguments it takes, what runs it, and the step a call to it records."""

    name: str
    step: ToolStep
    args_model: type[FrozenModel]
    run: Callable[..., Awaitable[tuple[RetrievedChunk, ...]]]
    description: str

    def definition(self) -> dict:
        """The tool as bind_tools wants it: an openai function-tool dictionary."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


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

TOOL_DEFINITIONS = [spec.definition() for spec in TOOL_SURFACE.values()]
"""The surface as the model is shown it, built once: it depends on nothing at call time."""


def tool_step(name: str) -> ToolStep:
    """The step a call to this tool records; a tool the surface does not have records that."""
    spec = TOOL_SURFACE.get(name)
    return spec.step if spec else ToolStep.UNKNOWN


async def run_tool_call(session: AsyncSession, call: ToolCall) -> tuple[RetrievedChunk, ...]:
    """One call's chunks; an unknown tool, an invalid target or a failing call yields
    nothing, never an error — the loop is best-effort and a bad call adds nothing."""
    spec = TOOL_SURFACE.get(call.name)
    if spec is None:
        logger.warning("assess called unknown tool %s", call.name)
        return ()
    try:
        args = spec.args_model.model_validate(call.args)
        return await spec.run(session, args)
    except ValidationError:
        logger.warning("assess called %s with invalid arguments", call.name)
        return ()
    except (LLMError, SQLAlchemyError) as exc:
        logger.warning("assess call to %s failed: %s", call.name, exc)
        return ()
