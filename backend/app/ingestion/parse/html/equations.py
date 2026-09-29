"""Equations EUR-Lex lays out as a one-row table, `label =` beside its expression, read as
one formula so the label and the maths around a drawn formula stay inside its delimiters."""

import re

from selectolax.parser import Node

from app.ingestion.parse.formex.latex import text_to_latex
from app.ingestion.parse.html.text import clean_text

FORMULA_RE = re.compile(r"\$\$.+?\$\$", re.DOTALL)
SUBSCRIPT = "oj-sub"
TEXT_NODE = "-text"


def _cell_latex(node: Node) -> str:
    """A cell as LaTeX: its text set upright, its subscripts as `_{}`, its formulas as written."""
    parts: list[str] = []
    for child in node.iter(include_text=True):
        if child.tag == TEXT_NODE:
            text = child.text_content or ""
            cursor = 0
            for match in FORMULA_RE.finditer(text):
                parts.append(text_to_latex(text[cursor : match.start()]))
                parts.append(match.group()[2:-2])
                cursor = match.end()
            parts.append(text_to_latex(text[cursor:]))
        elif SUBSCRIPT in (child.attributes.get("class") or "").split():
            parts.append(f"_{{{_cell_latex(child)}}}")
        else:
            parts.append(_cell_latex(child))
    return "".join(parts)


def equation_text(table: Node) -> str | None:
    """The whole equation as one `$$latex$$` where the table is a single `label =` row beside
    an expression holding a formula, None for any other table."""
    rows = table.css("tr")
    cells = rows[0].css("td, th") if len(rows) == 1 else []
    if not (
        len(cells) == 2
        and clean_text(cells[0].text()).endswith("=")
        and FORMULA_RE.search(cells[1].text())
    ):
        return None
    return f"$${' '.join(_cell_latex(rows[0]).split())}$$"
