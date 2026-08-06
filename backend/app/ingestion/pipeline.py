"""The ingest pipeline: one run, fetch -> parse -> chunk -> store."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.chunk.models import ChunkRunResult
from app.ingestion.chunk.stage import chunk_documents
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.fetch.service import get_other_topic_refs
from app.ingestion.fetch.stage import fetch_documents
from app.ingestion.models import StageRunResult
from app.ingestion.parse.models import ParseRunResult
from app.ingestion.parse.stage import parse_documents
from app.ingestion.schemas import IngestRun
from app.ingestion.service import complete_ingest_run, create_ingest_run

logger = logging.getLogger(__name__)


class IngestRunResult(BaseModel):
    """Outcome of one ingest run: one result per stage."""

    run_id: int
    corpus_version: str | None = None
    fetch: FetchRunResult = Field(default_factory=FetchRunResult)
    parse: ParseRunResult = Field(default_factory=ParseRunResult)
    chunk: ChunkRunResult = Field(default_factory=ChunkRunResult)

    @property
    def stages(self) -> dict[str, StageRunResult]:
        return {"fetch": self.fetch, "parse": self.parse, "chunk": self.chunk}

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.stages.values())

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.COMPLETED if self.ok else IngestRunStatus.FAILED

    def summary(self) -> str:
        """The run as the CLI prints it: a line per stage, then the per-ref detail."""
        return "\n".join(
            [
                f"run {self.run_id} ({self.corpus_version or 'not stamped'})",
                *(f"  [{name}] {result.summary()}" for name, result in self.stages.items()),
                *(
                    f"  {name} {line}"
                    for name, result in self.stages.items()
                    for line in result.details()
                ),
            ]
        )


async def _mark_failed(session: AsyncSession, run: IngestRun) -> None:
    """Discard the aborted transaction, then close the run out without masking why it aborted."""
    run_id = run.id
    await session.rollback()
    try:
        await complete_ingest_run(session, run, status=IngestRunStatus.FAILED)
    except SQLAlchemyError:
        logger.exception("run %s could not be marked failed", run_id)


@asynccontextmanager
async def ingest_run(session: AsyncSession) -> AsyncIterator[IngestRun]:
    """Open a run, and mark it failed if the pipeline raises or is interrupted."""
    run = await create_ingest_run(session)
    try:
        yield run
    except BaseException:
        await _mark_failed(session, run)
        raise


async def ingest(
    session: AsyncSession,
    *,
    client: httpx.Client,
    topics: Sequence[str],
    data_dir: Path,
) -> IngestRunResult:
    """Run the whole pipeline under one ingest run; blocking HTTP is fine here (CLI-only)."""
    async with ingest_run(session) as run:
        documents, fetched = await fetch_documents(
            session, client=client, topics=topics, data_dir=data_dir, run=run
        )
        logger.info("[fetch] %s", fetched.summary())

        parsed, parse_result = parse_documents(documents, data_dir=data_dir)
        logger.info("[parse] %s", parse_result.summary())

        keep = None
        if fetched.ok and parse_result.ok:
            keep = set(fetched.discovered) | await get_other_topic_refs(session, topics)

        chunked = await chunk_documents(session, parsed, ingest_run_id=run.id, keep_refs=keep)
        logger.info("[chunk] %s", chunked.summary())

        result = IngestRunResult(run_id=run.id, fetch=fetched, parse=parse_result, chunk=chunked)
        await complete_ingest_run(session, run, status=result.status)
        result.corpus_version = run.corpus_version
        return result
