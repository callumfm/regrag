from collections.abc import Callable

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.evals.models import UnresolvedReference
from app.evals.service import find_unresolved_references
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.schemas import IngestRun
from app.retrieval.models import ReferenceTarget
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
