# RegRag

Agentic RAG over EU maritime emissions regulation, with a built-in evaluation harness.

Monorepo: [backend/](backend/) (FastAPI) · [frontend/](frontend/) (React)

Local dev: `docker compose up -d db`, then `uv run fastapi dev` in [backend/](backend/) and `pnpm dev` in [frontend/](frontend/).

## Running the production stack locally

Day-to-day development runs only Postgres in Docker:

```bash
docker compose up -d
```

To bring up the full production topology — Caddy terminating TLS in front of the
API, with Postgres unpublished — copy the example environment file and use the
prod overlay:

```bash
cp .env.example .env.prod
docker compose --env-file .env.prod -f compose.yaml -f compose.prod.yaml up -d --build db
docker compose --env-file .env.prod -f compose.yaml -f compose.prod.yaml run --rm api alembic upgrade head
docker compose --env-file .env.prod -f compose.yaml -f compose.prod.yaml up -d

curl -k https://localhost/api/health
```

`DOMAIN=localhost` makes Caddy issue a certificate from its own internal CA, so
`-k` is expected. Setting `DOMAIN` to a real hostname switches the same Caddyfile
to Let's Encrypt with no other change.

The frontend is not served by this stack — it deploys to Cloudflare Pages and
reaches the API cross-origin via `CORS_ORIGINS`.
