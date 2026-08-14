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
        ("Directive", "92", "43", "31992L0043"),
        ("Regulation", "92", "2913", "31992R2913"),
    ],
)
def test_build_pads_the_number_and_maps_the_kind(kind, year, number, expected) -> None:
    assert celex.build(kind, year, number) == expected


def test_build_rejects_an_unknown_kind() -> None:
    with pytest.raises(KeyError):
        celex.build("Recommendation", "2015", "757")


@pytest.mark.parametrize("year", ["3021", "1", "757"])
def test_build_rejects_a_year_no_act_can_have(year: str) -> None:
    with pytest.raises(ValueError, match="not a legislation citation"):
        celex.build("Regulation", year, "757")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2015", 2015), ("92", 1992), ("87", 1987), ("757", None), ("3021", None), ("1805", None)],
)
def test_year_candidates_are_plausible_four_digit_years(value: str, expected: int | None) -> None:
    assert celex.as_year(value) == expected


@pytest.mark.parametrize(
    ("numbered", "pair", "expected", "citation"),
    [
        (True, ("765", "2008"), ("765", "2008"), "Regulation (EC) No 765/2008"),
        (True, ("2913", "92"), ("2913", "92"), "Regulation (EEC) No 2913/92"),
        (True, ("95", "93"), ("95", "93"), "Regulation (EEC) No 95/93"),
        (True, ("2003", "2003"), ("2003", "2003"), "Regulation (EC) No 2003/2003"),
        (True, ("2015", "757"), ("757", "2015"), "Regulation (EU) No 2015/757"),
        (True, ("2015", "1998"), ("1998", "2015"), "Regulation (EU) No 2015/1998"),
        (False, ("2015", "757"), ("757", "2015"), "Regulation (EU) 2015/757"),
        (False, ("2003", "87"), ("87", "2003"), "Directive 2003/87/EC"),
        (False, ("92", "43"), ("43", "92"), "Council Directive 92/43/EEC"),
        (False, ("1907", "2006"), ("1907", "2006"), "Regulation (EC) 1907/2006"),
        (False, ("2018", "2066"), ("2066", "2018"), "Regulation (EU) 2018/2066"),
    ],
)
def test_a_citation_pair_reads_as_number_then_year(numbered, pair, expected, citation) -> None:
    """A 'No' marks the old number-first scheme; without one, the likelier year is the year."""
    assert celex.order_number_and_year(numbered, *pair) == expected


def test_a_half_that_cannot_be_a_year_must_be_the_act_number() -> None:
    """2018/2066 only resolves because 2066 has not happened; as_year rejects future years."""
    assert celex.as_year("2066") is None
    assert celex.order_number_and_year(False, "2018", "2066") == ("2066", "2018")


@pytest.mark.parametrize("celex_id", ["32015R0757", "32003L0087", "32013D0162"])
def test_legislation_ids_are_recognised(celex_id: str) -> None:
    assert celex.is_legislation(celex_id)


@pytest.mark.parametrize(
    "celex_id",
    [
        "02015R0757-20250101",
        "52015PC0337",
        "62015CJ0001",
        "32015X0757",
        "3201",
        "392L0043",
    ],
)
def test_non_legislation_ids_are_rejected(celex_id: str) -> None:
    assert not celex.is_legislation(celex_id)


def test_consolidated_stem_swaps_the_sector_and_opens_the_date_suffix() -> None:
    assert celex.consolidated_stem("32015R0757") == "02015R0757-"


def test_consolidated_versions_share_their_act_stem() -> None:
    stem = celex.consolidated_stem("32015R0757")
    assert "02015R0757-20250101".startswith(stem)
    assert not "02023R1805-20250101".startswith(stem)
