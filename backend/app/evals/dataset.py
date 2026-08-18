"""The golden dataset: authored cases versioned as JSON beside this module."""

import hashlib
from pathlib import Path

from pydantic import TypeAdapter

from app.evals.models import EvalCase

GOLDEN_PATH = Path(__file__).with_name("golden.json")

_cases = TypeAdapter(tuple[EvalCase, ...])


def load_golden(path: Path = GOLDEN_PATH) -> tuple[EvalCase, ...]:
    """Every case in the file, validated."""
    return _cases.validate_json(path.read_bytes())


def dataset_hash(path: Path = GOLDEN_PATH) -> str:
    """The file's sha256, so a run records exactly which dataset it scored."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
