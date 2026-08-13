"""Shared test fixtures."""

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from tenacity import wait_none

from app.core.clock import utc_now
from app.core.config import EMBED_DIMENSIONS, R2Config, config
from app.core.db.session import async_session_factory
from app.core.storage import LocalObjectStore
from app.ingestion.chunk.models import Chunk
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.discover.models import DiscoveredDocument
from app.ingestion.discover.sparql import run_topic_query
from app.ingestion.embed.stage import _embed_batch
from app.ingestion.enums import IngestRunStatus, SectionKind
from app.ingestion.fetch.download import download_version
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.parse.html.parser import parse_eurlex_html
from app.ingestion.parse.models import ParsedDocument
from app.ingestion.schemas import IngestRun
from app.ingestion.storage import write_document
from app.main import configure_app

RETRIED = (run_topic_query, download_version, _embed_batch)

PARSE_FIXTURES = Path(__file__).parent / "ingestion" / "parse" / "fixtures"
FUELEU_HTML = (PARSE_FIXTURES / "32023R1805.html").read_text()
MRV_HTML = (PARSE_FIXTURES / "32015R0757.html").read_text()

R2_ENV = {
    "R2_ACCOUNT_ID": "acc",
    "R2_ACCESS_KEY_ID": "key",
    "R2_SECRET_ACCESS_KEY": "secret",
    "R2_BUCKET": "regrag-raw",
}


def r2_config(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> R2Config:
    """R2 settings as they really arrive — from the environment; an empty one is left unset."""
    for name, value in {**R2_ENV, **overrides}.items():
        monkeypatch.setenv(name, value)
    return R2Config()


@pytest.fixture
def defuse_retry(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable], Callable]:
    """Strip tenacity's waits from a retry-wrapped function, keeping its retry behaviour."""

    def _defuse(fn: Callable) -> Callable:
        # ty: ignore[unresolved-attribute] — tenacity sets .retry dynamically, untyped
        monkeypatch.setattr(fn.retry, "wait", wait_none())
        return fn

    return _defuse


@pytest.fixture(autouse=True)
def no_retry_backoff(defuse_retry: Callable[[Callable], Callable]) -> None:
    """Defuse every retry-wrapped callable in RETRIED, so retry tests don't sleep."""
    for fn in RETRIED:
        defuse_retry(fn)


@pytest.fixture(scope="session")
def db_engine() -> AsyncEngine:
    """NullPool so each test's connection lives and dies inside its own event loop."""
    return create_async_engine(config.SQLALCHEMY_DATABASE_URI, poolclass=NullPool)


@asynccontextmanager
async def rolled_back_session(
    db_engine: AsyncEngine, *, clear: bool = True
) -> AsyncGenerator[AsyncSession, None]:
    """Session bound to a transaction that is always rolled back, optionally clearing ingest tables.

    Savepoint join mode so a commit inside the session outlives a later rollback, as in production.
    """
    async with db_engine.connect() as conn:
        trans = await conn.begin()
        if clear:
            await conn.execute(delete(DocumentChunk))
            await conn.execute(delete(RawDocument))
            await conn.execute(delete(IngestRun))
        async with async_session_factory(
            bind=conn, join_transaction_mode="create_savepoint"
        ) as session:
            yield session
        await trans.rollback()


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """A session over empty ingest tables, whose writes never outlive the test."""
    async with rolled_back_session(db_engine) as session:
        yield session


@pytest.fixture
def make_document() -> Callable[..., RawDocument]:
    """Build a RawDocument whose identity fields derive from celex, overridable per field."""

    def _make(run: IngestRun, celex: str = "32023R1805", **overrides: Any) -> RawDocument:
        defaults: dict[str, Any] = {
            "run": run,
            "source": "eurlex",
            "celex": celex,
            "resolved_celex": celex,
            "topic": "fueleu",
            "url": f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}",
            "sha256": "a" * 64,
            "size_bytes": 758462,
            "fetched_at": utc_now(),
        }
        return RawDocument(**{**defaults, **overrides})

    return _make


@pytest.fixture
def local_store(tmp_path: Path) -> LocalObjectStore:
    """The real local store, rooted in tmp_path, so the suite needs no network."""
    return LocalObjectStore(tmp_path / "raw")


@pytest.fixture
def store_document(
    local_store: LocalObjectStore, make_document: Callable[..., RawDocument]
) -> Callable[..., RawDocument]:
    """Store bytes and return the row that keys them, so the row's sha is the stored one."""

    def _store(
        run: IngestRun,
        content: bytes = b"<html>act</html>",
        celex: str = "32023R1805",
        resolved_celex: str | None = None,
        **overrides: Any,
    ) -> RawDocument:
        resolved = resolved_celex or celex
        sha256, size_bytes = write_document(local_store, celex, resolved, content)
        return make_document(
            run,
            celex=celex,
            resolved_celex=resolved,
            sha256=sha256,
            size_bytes=size_bytes,
            **overrides,
        )

    return _store


@pytest.fixture(scope="session")
def fueleu() -> ParsedDocument:
    """The OJ dialect fixture, parsed once: a ParsedDocument is frozen, so tests share one."""
    return parse_eurlex_html(FUELEU_HTML, "32023R1805", "fueleu")


@pytest.fixture(scope="session")
def mrv() -> ParsedDocument:
    """The consolidated dialect fixture, parsed once and shared like fueleu."""
    return parse_eurlex_html(MRV_HTML, "32015R0757", "mrv")


@pytest.fixture
async def ingest_run(db_session: AsyncSession) -> IngestRun:
    """A flushed run for chunks to hang off, since their ingest_run_id is a real FK."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.fixture
async def later_run(db_session: AsyncSession, ingest_run: IngestRun) -> IngestRun:
    """A second run, ordered after ingest_run, for tests about which run a chunk belongs to."""
    run = IngestRun(status=IngestRunStatus.RUNNING)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.fixture
def make_chunk_row() -> Callable[..., DocumentChunk]:
    """Build a persisted-chunk row with sane defaults, overridable per field."""

    def _make(run: IngestRun, **overrides: Any) -> DocumentChunk:
        defaults: dict[str, Any] = {
            "run": run,
            "celex": "32023R1805",
            "topic": "fueleu",
            "content_hash": "b" * 64,
            "occurrence": 0,
            "kind": SectionKind.PARAGRAPH,
            "article": "4",
            "annex": None,
            "title": "Greenhouse gas intensity limit",
            "paragraph": "1",
            "heading_path": ["Chapter I", "Section 2"],
            "part": 1,
            "parts": 1,
            "position": 0,
            "citation": "Article 4(1)",
            "text": "The greenhouse gas intensity of the energy used on board.",
            "references": [{"raw": "Annex I", "annex": "I"}],
        }
        return DocumentChunk(**{**defaults, **overrides})

    return _make


@pytest.fixture
def app() -> FastAPI:
    """A throwaway app wired like production, so tests never mutate the real one."""
    app = FastAPI()
    configure_app(app)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class FakeProvider:
    """Records each batch's texts and answers with numbered vectors, raising on scripted calls."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.errors: dict[int, Exception] = {}

    async def __call__(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append(list(texts))
        if error := self.errors.get(len(self.calls)):
            raise error
        return [[float(index)] * EMBED_DIMENSIONS for index in range(len(texts))]


@pytest.fixture(autouse=True)
def embeddings(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    """No test reaches a provider: every embed call is recorded and answered locally."""
    provider = FakeProvider()
    monkeypatch.setattr("app.ingestion.embed.stage.embed", provider)
    return provider


@pytest.fixture
def corpus_client() -> Callable[..., tuple[httpx.AsyncClient, list[str]]]:
    """Transport serving SPARQL payloads per topic and HTML responses per celex."""

    def _make(
        sparql: dict[str, httpx.Response], docs: dict[str, httpx.Response]
    ) -> tuple[httpx.AsyncClient, list[str]]:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "publications.europa.eu":
                query = request.url.params["query"]
                for topic, seed in config.SEEDS.items():
                    if seed in query:
                        return sparql[topic]
                raise AssertionError(f"no seed in query: {query[:80]}")
            celex = request.url.params["uri"].removeprefix("CELEX:")
            calls.append(celex)
            return docs[celex]

        return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls

    return _make


def binding(celex: str, force: str | None = None, cons: str | None = None) -> dict:
    """One SPARQL result row for a celex, with optional in-force and consolidation."""
    b: dict = {"c": {"value": celex}}
    if force is not None:
        b["force"] = {"value": force}
    if cons is not None:
        b["cons"] = {"value": cons}
    return b


def payload(*bindings: dict) -> dict:
    """A SPARQL JSON response body wrapping the given result rows."""
    return {"results": {"bindings": list(bindings)}}


MRV_SPARQL = httpx.Response(
    200, json=payload(binding("32015R0757", force="1"), binding("32023R2449", force="1"))
)


def discovered_document(
    celex: str = "32015R0757", topic: str = "mrv", candidate: str | None = None
) -> DiscoveredDocument:
    """What discovery would hand fetch for one act, overridable per field."""
    return DiscoveredDocument(topic=topic, source="eurlex", celex=celex, candidate_celex=candidate)


async def chunk_versions(session: AsyncSession, celex: str | None = None) -> set[str | None]:
    """The corpus versions the stored chunks resolve to through the run that stored them."""
    stmt = select(IngestRun.corpus_version).join(
        DocumentChunk, DocumentChunk.ingest_run_id == IngestRun.id
    )
    if celex is not None:
        stmt = stmt.where(DocumentChunk.celex == celex)
    return set(await session.scalars(stmt))


async def chunk_rows(session: AsyncSession, celex: str | None = None) -> list[DocumentChunk]:
    """Persisted chunks in insertion order, for one document or the whole table."""
    stmt = select(DocumentChunk).order_by(DocumentChunk.id)
    if celex is not None:
        stmt = stmt.where(DocumentChunk.celex == celex)
    return list(await session.scalars(stmt))


def chunk(**overrides: Any) -> Chunk:
    """The chunker's value object with sane defaults, overridable per field."""
    defaults: dict[str, Any] = {
        "celex": "32023R1805",
        "topic": "fueleu",
        "kind": SectionKind.PARAGRAPH,
        "text": "The greenhouse gas intensity limit.",
        "article": "4",
        "paragraph": "1",
    }
    return Chunk(**{**defaults, **overrides})
