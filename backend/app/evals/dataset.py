"""The eval dataset: authored cases versioned as JSON beside this module."""

import hashlib
from pathlib import Path

from pydantic import TypeAdapter

from app.evals.models import EvalCase

CASES_PATH = Path(__file__).with_name("cases.json")

_cases = TypeAdapter(tuple[EvalCase, ...])


def load_cases(path: Path = CASES_PATH) -> tuple[EvalCase, ...]:
    """Every case in the file, validated."""
    return _cases.validate_json(path.read_bytes())


def dataset_hash(path: Path = CASES_PATH) -> str:
    """The file's sha256, so a run records exactly which dataset it scored."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
