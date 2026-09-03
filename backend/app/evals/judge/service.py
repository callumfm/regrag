"""Judging one case: the dimensions that apply to it, each one model call."""

import logging

import litellm
from pydantic import ValidationError

from app.chat.enums import ChatOutcome
from app.chat.models import ChatState
from app.core.config import config
from app.core.llm import LLMError, llm_retry, wrap_provider_errors
from app.core.models import FrozenModel
from app.evals.dataset.enums import EvalKind
from app.evals.dataset.models import EvalCase
from app.evals.judge.models import (
    CaseJudgement,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    RefusalVerdict,
)
from app.evals.judge.prompts import (
    CORRECTNESS_PROMPT,
    FAITHFULNESS_PROMPT,
    REFUSAL_PROMPT,
    build_correctness_message,
    build_faithfulness_message,
    build_refusal_message,
)
from app.evals.metrics import find_cited_markers

logger = logging.getLogger(__name__)


@llm_retry
@wrap_provider_errors("judge call")
async def call_judge_model[T: FrozenModel](system: str, user: str, output: type[T]) -> T:
    """One blocking judge turn answering in the verdict's own shape. A parameter the judge
    model does not accept is dropped rather than sent, so one call serves every provider;
    an answer off the schema is a failed call, not a verdict."""
    response = await litellm.acompletion(
        model=config.EVAL_JUDGE_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=output,
        api_key=config.ANTHROPIC_API_KEY.get_secret_value(),
        max_tokens=config.CHAT_MAX_TOKENS,
        timeout=config.CHAT_TIMEOUT,
        drop_params=True,
    )
    content = response.choices[0].message.content or ""
    try:
        return output.model_validate_json(content)
    except ValidationError as exc:
        logger.warning("judge answered off its schema: %s", exc)
        raise LLMError("judge call failed") from exc


async def _judge[T: FrozenModel](system: str, user: str, output: type[T]) -> T | None:
    """A dimension's verdict, or None when the call failed: a judge that cannot be reached
    leaves the dimension unmeasured rather than failing the run."""
    try:
        return await call_judge_model(system, user, output)
    except LLMError as exc:
        logger.warning("%s left unjudged: %s", output.__name__, exc)
        return None


async def judge_case(case: EvalCase, state: ChatState) -> CaseJudgement:
    """The dimensions this case can be judged on. Only an answered case is judged: a gate
    refusal is scored by the gate metrics, and an errored case is scored by nothing. An
    out-of-corpus answer is judged on whether it declined; an in-corpus answer on its
    correctness, and on its faithfulness when it cited anything to be faithful to."""
    if state.outcome is not ChatOutcome.DONE:
        return CaseJudgement()
    if case.kind is EvalKind.OUT_OF_CORPUS:
        message = build_refusal_message(case.question, state.answer)
        return CaseJudgement(refusal=await _judge(REFUSAL_PROMPT, message, RefusalVerdict))

    message = build_correctness_message(case.question, case.answer or "", state.answer)
    correctness = await _judge(CORRECTNESS_PROMPT, message, CorrectnessVerdict)
    faithfulness = None
    if find_cited_markers(state.answer):
        message = build_faithfulness_message(state.answer, state.sources)
        faithfulness = await _judge(FAITHFULNESS_PROMPT, message, FaithfulnessVerdict)
    return CaseJudgement(correctness=correctness, faithfulness=faithfulness)
