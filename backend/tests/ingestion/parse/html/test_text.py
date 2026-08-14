"""EUR-Lex text conventions: whitespace, amendment glyphs, emptied brackets."""

from app.ingestion.parse.html.text import clean_text


def test_clean_collapses_runs_of_whitespace():
    assert clean_text("a  \n   b\t c") == "a b c"


def test_clean_replaces_the_non_breaking_spaces_eurlex_indents_with():
    assert clean_text("1.\xa0\xa0\xa0The yearly average") == "1. The yearly average"


def test_clean_strips_leading_and_trailing_whitespace():
    assert clean_text("\n   ANNEX I\n   ") == "ANNEX I"


def test_clean_strips_amendment_markers_but_keeps_the_text_they_wrap():
    assert clean_text("▼M1 greenhouse gas emissions ◄") == "greenhouse gas emissions"


def test_clean_strips_the_brackets_an_emptied_footnote_leaves_behind():
    assert clean_text("Directive 2009/16/EC (  ) of the Council") == (
        "Directive 2009/16/EC of the Council"
    )


def test_clean_keeps_parentheses_that_still_have_content():
    assert clean_text("(a) ‘voyage’ means") == "(a) ‘voyage’ means"
