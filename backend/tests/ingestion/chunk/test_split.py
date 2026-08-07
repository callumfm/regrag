from app.ingestion.chunk.split import (
    cut_to_max_chars,
    pack_parts,
    split_section_text,
    split_table_rows,
    split_text,
)
from app.ingestion.enums import SectionKind
from app.ingestion.parse.models import Section


def test_cut_to_max_chars_yields_full_length_pieces_then_a_remainder() -> None:
    assert cut_to_max_chars("a" * 25, 10) == ["a" * 10, "a" * 10, "a" * 5]


def test_pack_parts_joins_until_the_next_part_would_overflow() -> None:
    assert pack_parts(["aaaa", "bbbb", "cccc"], 9, "\n") == ["aaaa\nbbbb", "cccc"]


def test_pack_parts_cuts_a_part_that_cannot_fit_on_its_own() -> None:
    assert pack_parts(["a" * 12], 5, "\n") == ["aaaaa", "aaaaa", "aa"]


def test_split_text_prefers_line_boundaries() -> None:
    assert split_text("aaaa\nbbbb\ncccc", 9) == ["aaaa\nbbbb", "cccc"]


def test_split_text_falls_back_to_sentence_boundaries_inside_an_overlong_line() -> None:
    assert split_text("One is here. Two is here. Three.", 26) == [
        "One is here. Two is here.",
        "Three.",
    ]


def test_split_text_drops_blank_lines() -> None:
    assert split_text("aaaa\n\nbbbb", 100) == ["aaaa\nbbbb"]


def test_split_text_stays_within_the_limit_whatever_the_text() -> None:
    assert max(len(piece) for piece in split_text("word " * 400, 100)) <= 100


def test_split_table_rows_repeats_the_header_on_every_piece() -> None:
    rows = (("Fuel", "Factor"),) + tuple((f"Fuel{n}", f"{n}.0") for n in range(12))
    pieces = split_table_rows(rows, 60)
    assert len(pieces) > 1
    assert all(piece.startswith("Fuel | Factor\n") for piece in pieces)


def test_split_table_rows_joins_cells_with_the_separator() -> None:
    assert split_table_rows((("Fuel", "Factor"), ("LNG", "2.75")), 100) == [
        "Fuel | Factor\nLNG | 2.75"
    ]


def test_split_table_rows_falls_back_to_text_when_the_header_leaves_no_budget() -> None:
    rows = (("A" * 30, "B" * 30), ("LNG", "2.75"))
    pieces = split_table_rows(rows, 20)
    assert max(len(piece) for piece in pieces) <= 20


def test_split_section_text_splits_a_table_section_on_its_rows() -> None:
    section = Section(kind=SectionKind.TABLE, rows=(("Fuel", "Factor"), ("LNG", "2.75")))
    assert split_section_text(section, 100) == ["Fuel | Factor\nLNG | 2.75"]


def test_split_section_text_yields_nothing_for_a_section_with_no_text() -> None:
    assert split_section_text(Section(kind=SectionKind.PARAGRAPH, text=""), 100) == []
