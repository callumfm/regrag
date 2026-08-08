"""The decisions the ingestion enumerations carry."""

from app.ingestion.enums import DocChange


def test_no_previous_run_makes_a_document_new():
    assert DocChange.between(None, "32023R2449") is DocChange.NEW


def test_a_differing_resolved_celex_is_changed():
    assert DocChange.between("02015R0757-20240101", "02015R0757-20250101") is DocChange.CHANGED


def test_the_same_resolved_celex_is_unchanged():
    assert DocChange.between("02015R0757-20250101", "02015R0757-20250101") is DocChange.UNCHANGED
