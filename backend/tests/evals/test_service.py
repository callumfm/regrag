from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.enums import EvalKind
from app.evals.models import EvalCase
from app.evals.service import unresolved_gold
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from app.retrieval.models import ReferenceTarget

pytestmark = pytest.mark.anyio

STORED = ReferenceTarget(celex="32023R1805", article="4")
STORED_ANNEX = ReferenceTarget(celex="32023R1805", annex="IV")
MISSING = ReferenceTarget(celex="32023R1805", article="999")


def case(id: str, kind: EvalKind = EvalKind.IN_CORPUS, *gold: ReferenceTarget) -> EvalCase:
    return EvalCase(
        id=id,
        kind=kind,
        question="q?",
        reference_answer="a" if kind is EvalKind.IN_CORPUS else None,
        gold=gold,
    )


async def test_gold_a_stored_chunk_answers_to_is_resolved(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ingest_run))
    db_session.add(
        make_chunk_row(
            ingest_run, article=None, annex="IV", citation="Annex IV", content_hash="c" * 64
        )
    )
    await db_session.flush()

    unresolved = await unresolved_gold(
        db_session, (case("ok", EvalKind.IN_CORPUS, STORED, STORED_ANNEX),)
    )

    assert unresolved == ()


async def test_gold_no_chunk_answers_to_is_reported_with_its_case(
    db_session: AsyncSession, ingest_run: IngestRun, make_chunk_row: Callable[..., DocumentChunk]
) -> None:
    db_session.add(make_chunk_row(ingest_run))
    await db_session.flush()

    unresolved = await unresolved_gold(
        db_session,
        (case("ok", EvalKind.IN_CORPUS, STORED), case("stale", EvalKind.IN_CORPUS, MISSING)),
    )

    assert unresolved == (("stale", MISSING),)


async def test_out_of_corpus_cases_have_nothing_to_resolve(db_session: AsyncSession) -> None:
    assert await unresolved_gold(db_session, (case("ooc", EvalKind.OUT_OF_CORPUS),)) == ()
