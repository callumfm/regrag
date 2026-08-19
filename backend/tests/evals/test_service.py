from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.enums import ChatNode
from app.chat.prompts import REFUSAL_ANSWER
from app.core.llm import LLMError
from app.evals import models, service
from app.evals.models import UnresolvedReference
from app.evals.service import find_unresolved_references
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from app.retrieval.models import ReferenceTarget
from tests.conftest import retrieved_chunk, search_result
from tests.evals.conftest import REFERENCE, eval_case, eval_dataset, out_of_corpus_case

pytestmark = pytest.mark.anyio

STORED_ANNEX = ReferenceTarget(celex="32023R1805", annex="IV")
MISSING = ReferenceTarget(celex="32023R1805", article="999")


async def test_a_reference_a_stored_chunk_answers_to_is_resolved(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ingest_run))
    db_session.add(
        make_chunk_row(
            ingest_run, article=None, annex="IV", citation="Annex IV", content_hash="c" * 64
        )
    )
    await db_session.flush()

    unresolved = await find_unresolved_references(
        db_session, eval_dataset(eval_case(references=(REFERENCE, STORED_ANNEX)))
    )

    assert unresolved == ()


async def test_a_reference_no_chunk_answers_to_is_reported_with_its_case(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ingest_run))
    await db_session.flush()

    unresolved = await find_unresolved_references(
        db_session,
        eval_dataset(eval_case(id="ok"), eval_case(id="stale", references=(MISSING,))),
    )

    assert unresolved == (UnresolvedReference(case_id="stale", target=MISSING),)


async def test_out_of_corpus_cases_have_nothing_to_resolve(db_session: AsyncSession) -> None:
    unresolved = await find_unresolved_references(db_session, eval_dataset(out_of_corpus_case()))

    assert unresolved == ()


@pytest.fixture
def fake_graph(monkeypatch):
    """Stand in for the compiled graph, yielding chosen node updates as astream would."""

    def _install(*updates: dict, raises: Exception | None = None):
        class _Graph:
            async def astream(self, state, stream_mode=None):
                if raises is not None:
                    raise raises
                for update in updates:
                    yield update

        monkeypatch.setattr(service, "chat_graph", _Graph())

    return _install


async def test_a_run_reads_hits_sources_answer_and_usage_off_the_updates(fake_graph):
    fake_graph(
        {ChatNode.RETRIEVE: {"hits": (search_result(),), "sources": (retrieved_chunk(),)}},
        {
            ChatNode.SYNTHESIZE: {
                "answer": "The limit applies [1].",
                "usage": {"input_tokens": 1200, "output_tokens": 180, "total_tokens": 1380},
            }
        },
    )

    result = await service.run_case(eval_case())

    assert result.expanded_recall == 1.0
    assert result.reference_citation_rate == 1.0
    assert (result.input_tokens, result.output_tokens) == (1200, 180)
    assert result.error is None


async def test_a_refused_case_carries_its_hits_even_though_nothing_reached_the_prompt(fake_graph):
    fake_graph(
        {ChatNode.RETRIEVE: {"hits": (search_result(),), "sources": ()}},
        {ChatNode.REFUSE: {"answer": REFUSAL_ANSWER}},
    )

    result = await service.run_case(eval_case())

    assert result.gate_refused
    assert result.refused_a_covered_case
    assert result.expanded_recall == 0.0


async def test_a_failing_case_is_recorded_as_an_error_not_raised(fake_graph):
    fake_graph(raises=LLMError("provider down"))

    result = await service.run_case(eval_case())

    assert result.error is not None
    assert "provider down" in result.error


async def test_a_run_scores_only_the_selected_cases_but_hashes_the_whole_dataset(fake_graph):
    fake_graph({ChatNode.SYNTHESIZE: {"answer": "a [1]."}})
    dataset = eval_dataset(eval_case(id="fueleu-one"), eval_case(id="mrv-two"))

    run = await service.run_dataset(dataset, pattern="fueleu")

    assert [r.case.id for r in run.results] == ["fueleu-one"]
    assert run.dataset_sha == dataset.sha256


def test_selecting_without_a_pattern_takes_every_case():
    dataset = eval_dataset(eval_case(id="a"), eval_case(id="b"))

    assert service.select_cases(dataset) == dataset.cases


async def test_an_unexpected_failure_keeps_its_type_and_detail(fake_graph, caplog):
    """`describe` answers an HTTP client, for whom every unexpected failure is one generic
    sentence. A run whose purpose is diagnosis must record the diagnosis instead."""
    fake_graph(raises=RuntimeError("connection pool exhausted"))

    result = await service.run_case(eval_case())

    assert result.error == "RuntimeError: connection pool exhausted"
    assert "An unexpected error occurred" not in (result.error or "")
    assert "eval case case failed unexpectedly" in caplog.text
    assert "RuntimeError: connection pool exhausted" in caplog.text


async def test_a_domain_failure_records_the_message_it_speaks_for_itself(fake_graph):
    fake_graph(raises=LLMError("provider down"))

    result = await service.run_case(eval_case())

    assert result.error == "provider down"


async def test_a_usage_payload_missing_its_counts_costs_no_case(fake_graph):
    """The tokens are unmeasured; a KeyError here would unwind the whole run."""
    fake_graph({ChatNode.SYNTHESIZE: {"answer": "a [1].", "usage": {"total_tokens": 12}}})

    result = await service.run_case(eval_case())

    assert result.error is None
    assert (result.input_tokens, result.output_tokens) == (None, None)


async def test_a_run_stamps_the_instant_it_started_not_the_one_it_finished(fake_graph, monkeypatch):
    """The file is named for when the run began, so it lines up with that run's logs."""
    started = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(service, "utc_now", lambda: started)
    monkeypatch.setattr(models, "utc_now", lambda: datetime(2026, 8, 18, 12, 0, 55, tzinfo=UTC))
    fake_graph({ChatNode.SYNTHESIZE: {"answer": "a [1]."}})

    run = await service.run_dataset(eval_dataset(eval_case()))

    assert run.started_at == started


async def test_a_run_records_the_pattern_that_chose_its_cases(fake_graph):
    """The sha covers the whole dataset, so without the pattern a subset run reads as full."""
    fake_graph({ChatNode.SYNTHESIZE: {"answer": "a [1]."}})
    dataset = eval_dataset(eval_case(id="fueleu-one"), eval_case(id="mrv-two"))

    run = await service.run_dataset(dataset, pattern="fueleu")

    assert run.case_pattern == "fueleu"
    assert (await service.run_dataset(dataset)).case_pattern is None
