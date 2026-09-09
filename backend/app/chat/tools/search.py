"""search: a fresh corpus search, for a concept the context names without citing."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ToolStep
from app.chat.tools.models import ToolSpec
from app.core.config import config
from app.core.models import FrozenModel
from app.retrieval.models import RetrievedChunk, SearchFilters, SearchRequest
from app.retrieval.search import search
from app.retrieval.thresholds import meets_thresholds


class SearchArgs(FrozenModel):
    """A fresh corpus search: for what is named in the context but not cited."""

    query: str
    celex: str | None = None


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


SEARCH = ToolSpec(
    name="search",
    step=ToolStep.SEARCH,
    args_model=SearchArgs,
    run=run_search,
    description="Search the corpus for text matching a query, optionally within one act (celex). "
    "Use for concepts the context names without citing, or question parts with no "
    "context at all.",
)
