#! /usr/bin/env bash

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT/backend"
uv run python -c "import app.main; import json; print(json.dumps(app.main.app.openapi()))" > "$REPO_ROOT/frontend/openapi.json"

cd "$REPO_ROOT/frontend"
pnpm generate-api
