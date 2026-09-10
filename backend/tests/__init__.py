"""Test package: pin the environment and the database before any app import loads config."""

import os

os.environ["ENVIRONMENT"] = "test"
os.environ["DB_NAME"] = "regrag_test"
"""Set here rather than in .env.example, which is also the template for .env.dev: naming the
test database there would point a fresh dev checkout at the one the suite truncates."""
