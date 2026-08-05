# RegRag

Agentic RAG over EU maritime emissions regulation, with a built-in evaluation harness.

Monorepo: [backend/](backend/) (FastAPI) · [frontend/](frontend/) (React)

Local dev: `docker compose up -d db`, then `uv run fastapi dev` in [backend/](backend/) and `pnpm dev` in [frontend/](frontend/).

## Running the production stack locally

To bring up the full production topology — Caddy terminating TLS in front of the
API, with Postgres unpublished — write a `.env.prod` (gitignored) at the repo
root and use the prod overlay:

```bash
cat > .env.prod <<'EOF'
DOMAIN=localhost
ACME_EMAIL=dev@example.com
DB_PASS=postgres
CORS_ORIGINS=["http://localhost:5173"]
EOF

export COMPOSE_FILE=compose.yaml:compose.prod.yaml COMPOSE_ENV_FILES=.env.prod

docker compose run --build --rm api alembic upgrade head
docker compose up -d --wait

curl -k https://localhost/api/health
```

`DOMAIN=localhost` makes Caddy issue a certificate from its own internal CA, so
`-k` is expected. Setting `DOMAIN` to a real hostname switches the same Caddyfile
to Let's Encrypt with no other change.

Only deployment choices live in `.env.prod`; the overlay sets `ENVIRONMENT`,
`DB_HOST`, `ROOT_PATH`, and `RAW_DATA_DIR` itself, since those are properties of
the topology. Compose fails loudly and names any required variable you omit.

The frontend is not served by this stack — it deploys to Cloudflare Pages and
reaches the API cross-origin via `CORS_ORIGINS`.
