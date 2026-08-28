# Backend

FastAPI backend for RegRag: the ingestion pipeline, the retrieval layer, and the
chat API that answers from them.

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
| `app/chat/`      | The answering graph, its SSE endpoint, and the request ledger |
| `app/evals/`     | The golden dataset and the runner that scores the graph on it |
| `migrations/`    | Alembic revisions                                             |
| `tests/`         | Mirrors `app/`, with shared fixtures in `tests/conftest.py`   |

`app/core/` holds only what two or more capabilities use. Anything used by one
capability lives in that capability's package.

Each capability package follows the same file convention:

| File          | Purpose                                  |
| ------------- | ---------------------------------------- |
| `schemas.py`  | SQLAlchemy ORM models                    |
| `models.py`   | Pydantic request/response/value models   |
| `enums.py`    | Enumerations the capability owns         |
| `service.py`  | Database reads and writes                |
| `pipeline.py` | Orchestration across stages              |
| `router.py`   | FastAPI endpoints (where applicable)     |
| `cli.py`      | argparse entry point, listed in `[project.scripts]` |

New ORM schemas must be imported in `app/core/db/registry.py` so their mappers
register; a guard test fails if one is missing.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/),
[pre-commit](https://pre-commit.com/#install), Docker running.

```bash
uv sync
pre-commit install          # from the repo root
cp .env.example .env.dev    # then set VOYAGE_API_KEY and ANTHROPIC_API_KEY
```

Start the database, migrate and run the API:

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

## Chat

`POST /chat` streams a cited answer over SSE: one `sources` frame naming the
context blocks the answer may cite, `text` frames as the model writes, then
`done` — or a single `error` frame. A question the corpus does not cover is
refused before any model call, and every request is recorded with the path it
took through the graph, its timings and its tokens.

It calls the answering model, so it needs `ANTHROPIC_API_KEY`. Which model, and
how much context it is given, are the `CHAT_*` settings in `app/core/config.py`.

## Evals

`app/evals/dataset/golden.json` holds authored cases: a question, what a right
answer must say, and the articles it must come from — plus out-of-corpus
questions the system is expected to refuse. Each cited article is stamped with
the chunks it held when the case was authored, and the file records the corpus
version behind those stamps.

```bash
uv run evals check              # what no longer resolves, what moved, what is unstamped
uv run evals stamp              # record what the cited text says now
uv run evals stamp --case mrv   # ...for these cases only, leaving the rest stale
uv run evals run                # score the dataset against the current graph
uv run evals run --verbose      # ...listing every case with its own scores
uv run evals run --case fueleu  # only cases whose id contains this
uv run evals run --no-cache     # pay for every embed and rerank again
```

A run drives the same graph the endpoint runs, one case at a time so a timing
measures one case, and prints the settings it ran under beside the scores:
what search found before and after section expansion, what the answers cited,
what the refusal gate caught, and what the run cost. It exits non-zero if any
case raised.

Stamps catch the drift a resolving reference hides: an amendment rewrites
Article 4, the chunk still retrieves, and the case scores green against a
reference answer that is now wrong. `check` and `run` both name such a case,
neither fails on it — repairing one means a human re-reading the new text and
rewriting the answer, then `evals stamp` to record that it was read. Stamps are
left out of `dataset_sha`, so a re-stamp keeps past runs comparable.

Replayed embed and rerank calls are cached under `data/cache/evals`.
