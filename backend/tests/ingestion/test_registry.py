"""Registry invariants: the corpus is well-formed and stable."""

import dataclasses
import re

import pytest

from app.ingestion.registry import CORPUS, DocumentSpec

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
CELEX_PATTERN = re.compile(r"^3\d{4}[A-Z]\d{4}$")


def test_corpus_has_17_entries():
    assert len(CORPUS) == 17


def test_names_are_unique_slugs():
    names = [spec.name for spec in CORPUS]
    assert len(set(names)) == len(names)
    assert all(SLUG_PATTERN.match(name) for name in names)


def test_refs_are_unique_base_celex_ids():
    refs = [spec.ref for spec in CORPUS]
    assert len(set(refs)) == len(refs)
    assert all(CELEX_PATTERN.match(ref) for ref in refs)


def test_all_sources_are_eurlex():
    assert all(spec.source == "eurlex" for spec in CORPUS)


def test_spec_is_frozen():
    spec: DocumentSpec = CORPUS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "changed"  # type: ignore
