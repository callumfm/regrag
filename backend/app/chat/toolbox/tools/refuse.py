"""refuse: assess's word that nothing in the context bears on the question, and no fetch
of this corpus would."""

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import RefusalReason, ToolStep
from app.chat.models import Refusal
from app.chat.toolbox.models import ToolCall, ToolSpec
from app.core.models import FrozenModel
from app.retrieval.models import RetrievedChunk


class RefuseArgs(FrozenModel):
    """Assess's word that nothing in the context bears on the question and no fetch would:
    the one call that ends the question in the fixed refusal rather than an answer."""

    explanation: str


async def run_refusal(session: AsyncSession, args: RefuseArgs) -> tuple[RetrievedChunk, ...]:
    """Nothing to fetch: the call is its own result, and the round reads the explanation off
    it. Run like the others so the round records it as a step, timed and named."""
    return ()


REFUSE = ToolSpec(
    name="refuse",
    step=ToolStep.REFUSE,
    args_model=RefuseArgs,
    run=run_refusal,
    description="Refuse the question, because nothing in the context bears on it and no search "
    "or fetch of this corpus could change that. Call it alone, never beside a "
    "search or fetch.",
)


def is_refusal(call: ToolCall) -> bool:
    """Whether the call is assess's word that the context cannot answer, rather than a fetch."""
    return call.name == REFUSE.name


def refusal_from(call: ToolCall) -> Refusal | None:
    """The refusal a refuse call carries, its explanation read through the tool's arguments;
    an explanation the model left off or malformed is recorded as empty. None for a fetch."""
    if not is_refusal(call):
        return None
    try:
        explanation = RefuseArgs.model_validate(call.args).explanation
    except ValidationError:
        explanation = ""
    return Refusal(reason=RefusalReason.INSUFFICIENT_CONTEXT, explanation=explanation)
