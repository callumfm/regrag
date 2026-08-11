# Backend

FastAPI backend for RegRag: the ingestion pipeline, the retrieval agent and the evaluation harness.

## Stack

| Component              | Technology                                                    |
| ---------------------- | ------------------------------------------------------------- |
| Web Framework          | [FastAPI](https://github.com/fastapi/fastapi)                 |
| Database               | [PostgreSQL](https://www.postgresql.org/)                     |
| Vector Search          | [pgvector](https://github.com/pgvector/pgvector)              |
| Object Storage         | [Cloudflare R2](https://developers.cloudflare.com/r2/)        |
| ORM                    | [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy)        |
| Database Migrations    | [Alembic](https://github.com/sqlalchemy/alembic)              |
| Data Validation        | [Pydantic](https://github.com/pydantic/pydantic)              |
| LLM Gateway            | [LiteLLM](https://github.com/BerriAI/litellm)                 |
| Embeddings             | [Voyage](https://docs.voyageai.com/)                          |
| HTML Parsing           | [selectolax](https://github.com/rushter/selectolax)           |
| Containerisation       | [Docker](https://www.docker.com)                              |
| Python Package Manager | [uv](https://github.com/astral-sh/uv)                         |
| Linter                 | [Ruff](https://github.com/astral-sh/ruff)                     |
| Type Checker           | [ty](https://github.com/astral-sh/ty)                         |

## Layout

| Directory        | Contents                                                      |
| ---------------- | ------------------------------------------------------------- |
| `app/core/`      | Shared contracts: config, db session, http, llm, storage      |
| `app/ingestion/` | The corpus pipeline: discover → fetch → parse → chunk → embed |
| `app/retrieval/` | The read side: hybrid search and exact article lookup         |
| `migrations/`    | Alembic revisions                                             |
| `tests/`         | Mirrors `app/`, with shared fixtures in `tests/conftest.py`   |

`app/core/` holds only what two or more capabilities use. Anything used by one
capability lives in that capability's package.

Each capability package follows the same file convention:

| File          | Purpose                                  |
| ------------- | ---------------------------------------- |
| `schemas.py`  | SQLAlchemy ORM models                    |
| `models.py`   | Pydantic request/response/value models   |
| `service.py`  | Database reads and writes                |
| `pipeline.py` | Orchestration across stages              |
| `router.py`   | FastAPI endpoints (where applicable)     |

New ORM schemas must be imported in `app/core/db/registry.py` so their mappers
register; a guard test fails if one is missing.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/),
[pre-commit](https://pre-commit.com/#install), Docker running.

```bash
uv sync
pre-commit install          # from the repo root
cp .env.example .env.dev    # then set VOYAGE_API_KEY
```

Start the database from the repo root, then migrate and run the API:

```bash
docker compose up -d db
uv run alembic upgrade head
uv run fastapi dev
```

The API is then on `http://localhost:8000`, with `/health` reporting database
connectivity and `/docs` serving the OpenAPI schema.

## Ingest

```bash
uv run ingest             # every seed topic
uv run ingest fueleu      # one topic
```

A run stores the documents it downloads and calls the embedding model, so it
needs `VOYAGE_API_KEY` set. Re-running is cheap and safe: unchanged documents
are neither downloaded nor re-embedded.

Where the documents are stored depends on `STORAGE_BACKEND`. It defaults to
`local`, writing files under `RAW_DATA_DIR` (`<repo>/data/raw`), so dev and
tests need no network. Set it to `r2` and fill in the `R2_*` variables to use
the Cloudflare bucket instead.

What each stage does, and why, is in
[`app/ingestion/README.md`](app/ingestion/README.md).
