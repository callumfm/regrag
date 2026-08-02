#! /usr/bin/env bash

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT/backend"
uv run python -m scripts.export_openapi

cd "$REPO_ROOT/frontend"
pnpm generate-api
