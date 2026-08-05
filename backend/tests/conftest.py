"""Shared test fixtures."""

from collections.abc import AsyncGenerator, Callable
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from tenacity import wait_none

from app.core.clock import utc_now
from app.core.config import config
from app.core.db.session import async_session_factory
from app.core.http import PACE_SECONDS
from app.ingestion.chunk.models import Chunk
from app.ingestion.chunk.schemas import DocumentChunk
from app.ingestion.constants import SEEDS
from app.ingestion.enums import SectionKind
from app.ingestion.fetch import stage
from app.ingestion.fetch.discover import discover
from app.ingestion.fetch.resolve import resolve_version
from app.ingestion.fetch.schemas import RawDocument
from app.ingestion.fetch.stage import download
from app.ingestion.schemas import IngestRun
from app.main import configure_app

RETRIED = (discover, resolve_version, download)


@pytest.fixture
def defuse_retry(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable], Callable]:
    """Strip tenacity's waits from a @transient_retry function, keeping its retry behaviour."""

    def _defuse(fn: Callable) -> Callable:
        # ty: ignore[unresolved-attribute] — tenacity sets .retry dynamically, untyped
        monkeypatch.setattr(fn.retry, "wait", wait_none())
        return fn

    return _defuse


@pytest.fixture(autouse=True)
def no_retry_backoff(defuse_retry: Callable[[Callable], Callable]) -> None:
    """Defuse every @transient_retry callable in RETRIED, so retry tests don't sleep."""
    for fn in RETRIED:
        defuse_retry(fn)


@pytest.fixture(scope="session")
def db_engine() -> AsyncEngine:
    """NullPool so each test's connection lives and dies inside its own event loop."""
    return create_async_engine(config.SQLALCHEMY_DATABASE_URI, poolclass=NullPool)


@pytest.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Session bound to a transaction that is always rolled back, with ingest tables cleared."""
    async with db_engine.connect() as conn:
        trans = await conn.begin()
        await conn.execute(delete(DocumentChunk))
        await conn.execute(delete(RawDocument))
        await conn.execute(delete(IngestRun))
        async with async_session_factory(bind=conn) as session:
            yield session
        await trans.rollback()


@pytest.fixture
def make_document() -> Callable[..., RawDocument]:
    """Build a RawDocument whose identity fields derive from ref, overridable per field."""

    def _make(run: IngestRun, ref: str = "32023R1805", **overrides: Any) -> RawDocument:
        defaults: dict[str, Any] = {
            "run": run,
            "source": "eurlex",
            "ref": ref,
            "resolved_ref": ref,
            "topic": "fueleu",
            "url": f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{ref}",
            "sha256": "a" * 64,
            "size_bytes": 758462,
            "fetched_at": utc_now(),
        }
        return RawDocument(**{**defaults, **overrides})

    return _make


@pytest.fixture
def make_chunk_row() -> Callable[..., DocumentChunk]:
    """Build a persisted-chunk row with sane defaults, overridable per field."""

    def _make(**overrides: Any) -> DocumentChunk:
        defaults: dict[str, Any] = {
            "ref": "32023R1805",
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
            "citation": "Article 4(1)",
            "text": "The greenhouse gas intensity of the energy used on board.",
            "references": [{"raw": "Annex I", "annex": "I"}],
            "corpus_version": "2026-08-05-a3f9e21",
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


@pytest.fixture(autouse=True)
def paces(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Count pacing delays instead of sleeping through them."""
    calls: list[float] = []
    monkeypatch.setattr(stage, "pace", lambda: calls.append(PACE_SECONDS))
    return calls


@pytest.fixture
def corpus_client() -> Callable[..., tuple[httpx.Client, list[str]]]:
    """Transport serving SPARQL payloads per topic and HTML responses per celex ref."""

    def _make(
        sparql: dict[str, httpx.Response], docs: dict[str, httpx.Response]
    ) -> tuple[httpx.Client, list[str]]:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "publications.europa.eu":
                query = request.url.params["query"]
                for topic, seed in SEEDS.items():
                    if seed in query:
                        return sparql[topic]
                raise AssertionError(f"no seed in query: {query[:80]}")
            celex = request.url.params["uri"].removeprefix("CELEX:")
            calls.append(celex)
            return docs[celex]

        return httpx.Client(transport=httpx.MockTransport(handler)), calls

    return _make


def binding(celex: str, force: str | None = None, cons: str | None = None) -> dict:
    """One SPARQL result row for a celex ref, with optional in-force and consolidation."""
    b: dict = {"c": {"value": celex}}
    if force is not None:
        b["force"] = {"value": force}
    if cons is not None:
        b["cons"] = {"value": cons}
    return b


def payload(*bindings: dict) -> dict:
    """A SPARQL JSON response body wrapping the given result rows."""
    return {"results": {"bindings": list(bindings)}}


def chunk(**overrides: Any) -> Chunk:
    """The chunker's value object with sane defaults, overridable per field."""
    defaults: dict[str, Any] = {
        "ref": "32023R1805",
        "topic": "fueleu",
        "kind": SectionKind.PARAGRAPH,
        "text": "The greenhouse gas intensity limit.",
        "article": "4",
        "paragraph": "1",
    }
    return Chunk(**{**defaults, **overrides})
