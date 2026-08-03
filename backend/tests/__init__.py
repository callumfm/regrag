"""Test package: pin the environment before any app import loads config."""

import os

os.environ["ENVIRONMENT"] = "test"
