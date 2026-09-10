"""Test package: pin the environment and the database before any app import loads config."""

import os

os.environ["ENVIRONMENT"] = "test"
os.environ["DB_NAME"] = "regrag_test"
"""Set here rather than in .env.example, which is also the template for .env.dev: naming the
test database there would point a fresh dev checkout at the one the suite truncates."""

PLACEHOLDER_KEY = "test-not-a-real-key"
"""The entry points refuse to start without a key for every model they call, and .env.example
leaves both empty so a stray unmocked call cannot reach a provider. This satisfies the check
and still buys nothing but a 401."""

os.environ.setdefault("ANTHROPIC_API_KEY", PLACEHOLDER_KEY)
os.environ.setdefault("VOYAGE_API_KEY", PLACEHOLDER_KEY)
