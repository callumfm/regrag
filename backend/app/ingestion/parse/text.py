"""Format-neutral EUR-Lex text conventions, shared by every dialect and every source format."""

import re

FORMULA_PLACEHOLDER = "[formula]"

WHITESPACE_RE = re.compile(r"\s+")
AMENDMENT_GLYPHS = "▼►◄"
AMENDMENT_MARKER_RE = re.compile(r"[▼►]\s*[A-Z]+\d*|◄")
EMPTY_PARENS_RE = re.compile(r"\s*\(\s*\)")

ARTICLE_NUMBER_RE = re.compile(r"Article\s+(\d+[a-z]?)", re.IGNORECASE)
ANNEX_NUMBER_RE = re.compile(r"ANNEX\s+([IVXLC]+|\d+)", re.IGNORECASE)
LEADING_NUMBER_RE = re.compile(r"^(\d+[a-z]?)\.\s*")


def normalise_whitespace(text: str) -> str:
    """Collapse whitespace, including the non-breaking spaces EUR-Lex indents with."""
    return WHITESPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def clean_text(text: str) -> str:
    """Normalise whitespace, dropping amendment glyphs and the brackets that
    drop_non_legal_markup leaves behind when it empties a footnote reference.
    """
    if any(glyph in text for glyph in AMENDMENT_GLYPHS):
        text = AMENDMENT_MARKER_RE.sub(" ", text)
    if "(" in text:
        text = EMPTY_PARENS_RE.sub("", text)
    return normalise_whitespace(text)
