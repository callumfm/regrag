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
from app.ingestion.embed.models import EmbedRunResult
from app.ingestion.embed.stage import embed_chunks
from app.ingestion.enums import IngestRunStatus
from app.ingestion.fetch.models import FetchRunResult
from app.ingestion.fetch.service import get_other_topic_celexes
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
    embed: EmbedRunResult = Field(default_factory=EmbedRunResult)

    @property
    def stages(self) -> dict[str, StageRunResult]:
        return {
            "fetch": self.fetch,
            "parse": self.parse,
            "chunk": self.chunk,
            "embed": self.embed,
        }

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.stages.values())

    @property
    def status(self) -> IngestRunStatus:
        return IngestRunStatus.COMPLETED if self.ok else IngestRunStatus.FAILED

    def summary(self) -> str:
        """The run as the CLI prints it: a line per stage, then the per-celex detail."""
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


async def _known_corpus_celexes(
    session: AsyncSession,
    *,
    fetch_result: FetchRunResult,
    parse_result: ParseRunResult,
    topics: Sequence[str],
) -> set[str] | None:
    """The celexes the corpus consists of after this run: what it discovered, plus other topics'.

    None when any stage failed: pruning is irreversible, so a run with errors does not earn it.
    """
    if not (fetch_result.ok and parse_result.ok):
        return None
    return set(fetch_result.discovered) | await get_other_topic_celexes(session, topics)


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
        documents, fetch_result = await fetch_documents(
            session, client=client, topics=topics, data_dir=data_dir, run=run
        )
        logger.info("[fetch] %s", fetch_result.summary())

        parsed, parse_result = parse_documents(documents, data_dir=data_dir)
        logger.info("[parse] %s", parse_result.summary())

        corpus_celexes = await _known_corpus_celexes(
            session, fetch_result=fetch_result, parse_result=parse_result, topics=topics
        )
        chunk_result = await chunk_documents(
            session, parsed, ingest_run_id=run.id, corpus_celexes=corpus_celexes
        )
        logger.info("[chunk] %s", chunk_result.summary())

        embed_result = await embed_chunks(session)
        logger.info("[embed] %s", embed_result.summary())

        result = IngestRunResult(
            run_id=run.id,
            fetch=fetch_result,
            parse=parse_result,
            chunk=chunk_result,
            embed=embed_result,
        )
        await complete_ingest_run(session, run, status=result.status)
        result.corpus_version = run.corpus_version
        return result
