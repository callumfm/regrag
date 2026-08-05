"""CELEX id format: recognition, assembly and the consolidated-version stem."""

import pytest

from app.ingestion import celex


@pytest.mark.parametrize(
    ("kind", "year", "number", "expected"),
    [
        ("Regulation", "2015", "757", "32015R0757"),
        ("Directive", "2003", "87", "32003L0087"),
        ("Decision", "2013", "162", "32013D0162"),
        ("regulation", "2008", "765", "32008R0765"),
        ("Regulation", "2023", "1805", "32023R1805"),
    ],
)
def test_build_pads_the_number_and_maps_the_kind(kind, year, number, expected) -> None:
    assert celex.build(kind, year, number) == expected


def test_build_rejects_an_unknown_kind() -> None:
    with pytest.raises(KeyError):
        celex.build("Recommendation", "2015", "757")


@pytest.mark.parametrize("ref", ["32015R0757", "32003L0087", "32013D0162"])
def test_legislation_ids_are_recognised(ref: str) -> None:
    assert celex.is_legislation(ref)


@pytest.mark.parametrize(
    "ref",
    [
        "02015R0757-20250101",
        "52015PC0337",
        "62015CJ0001",
        "32015X0757",
        "3201",
    ],
)
def test_non_legislation_ids_are_rejected(ref: str) -> None:
    assert not celex.is_legislation(ref)


def test_consolidated_stem_swaps_the_sector_and_opens_the_date_suffix() -> None:
    assert celex.consolidated_stem("32015R0757") == "02015R0757-"


def test_consolidated_versions_share_their_act_stem() -> None:
    stem = celex.consolidated_stem("32015R0757")
    assert "02015R0757-20250101".startswith(stem)
    assert not "02023R1805-20250101".startswith(stem)
