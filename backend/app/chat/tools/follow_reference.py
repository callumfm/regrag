"""follow_reference: the full text of a division the context cites, fetched by its address."""

from collections.abc import Sequence

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ToolStep
from app.chat.tools.models import ToolCall, ToolSpec
from app.core.config import config
from app.retrieval.follow import follow_reference
from app.retrieval.models import ReferenceTarget, RetrievedChunk


async def run_follow_reference(
    session: AsyncSession, target: ReferenceTarget
) -> tuple[RetrievedChunk, ...]:
    """The division's text from the top, capped: a long article or annex would otherwise
    spend the round's whole budget on one call. No score to gate on — the context cited it."""
    chunks = await follow_reference(session, target)
    return chunks[: config.ASSESS_FOLLOW_LIMIT]


def already_in_context(call: ToolCall, sources: Sequence[RetrievedChunk]) -> bool:
    """Whether the call would only fetch a paragraph the context already shows in full, so
    running it could add nothing. A whole article or annex is never known to be shown in
    full: its chapeau's parts say nothing about what sits under it. A call the surface cannot
    read is left to run_tool_call to reject."""
    if call.name != FOLLOW_REFERENCE.name:
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


FOLLOW_REFERENCE = ToolSpec(
    name="follow_reference",
    step=ToolStep.FOLLOW_REFERENCE,
    args_model=ReferenceTarget,
    run=run_follow_reference,
    description="Fetch the full text of one cited division: an article (optionally one paragraph) "
    "or an annex of an act (celex). Use the addresses on the context's cites lines.",
)
