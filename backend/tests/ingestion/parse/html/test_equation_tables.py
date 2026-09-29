"""Equations EUR-Lex lays out as a one-row table, from the FuelEU annexes as they are in prod."""

import pytest

from app.ingestion.enums import SectionKind
from app.ingestion.parse.html.document import parse_eurlex_html
from tests.ingestion.parse.html.helpers import all_sections, annexes

IMAGE = '<img src="data:image/png;base64,AAAA"/>'
COMPLIANCE_BALANCE_SUM = r"$$\sum_{i}^{\text{nfuel}}M_{i} \times \text{LCV}_{i}$$"
RFNBO_EXPRESSION = r"$$0.02 \times \sum_{i}M_{i}$$"


def annex_with(body: str) -> str:
    return (
        "<html><body>"
        '<div class="eli-subdivision" id="art_1"><p class="oj-ti-art">Article 1</p>'
        '<p class="oj-normal">Subject matter.</p></div>'
        '<div class="eli-container" id="anx_IV"><p class="oj-doc-ti">ANNEX IV</p>'
        f'<p class="oj-doc-ti">Formulas</p>{body}</div>'
        "</body></html>"
    )


def one_row_table(*cells: str) -> str:
    tds = "\n".join(f'<td class="oj-table"><p class="oj-tbl-txt">{cell}</p></td>' for cell in cells)
    return f'<table class="oj-table"><tbody><tr class="oj-table">{tds}</tr></tbody></table>'


@pytest.mark.parametrize(
    ("body", "formulas", "children"),
    [
        pytest.param(
            '<p class="oj-normal">The following formula shall apply:</p>'
            + one_row_table(
                'Compliance balance [gCO<span class="oj-sub">2eq</span>] =',
                "x [<figure>" + IMAGE + "</figure> ]",
            )
            + '<p class="oj-normal">Where:</p>',
            (COMPLIANCE_BALANCE_SUM,),
            [
                (SectionKind.PARAGRAPH, "The following formula shall apply:"),
                (
                    SectionKind.PARAGRAPH,
                    r"$$\text{Compliance balance} [\text{gCO}_{2\text{eq}}] = x ["
                    r"\sum_{i}^{\text{nfuel}}M_{i} \times \text{LCV}_{i} ]$$",
                ),
                (SectionKind.PARAGRAPH, "Where:"),
            ],
            id="the compliance balance: label, text around the drawn sum and its subscripts "
            "become one formula on its own line",
        ),
        pytest.param(
            one_row_table('CB<span class="oj-sub">RFNBO</span>[MJ] =', IMAGE),
            (RFNBO_EXPRESSION,),
            [
                (
                    SectionKind.PARAGRAPH,
                    r"$$\text{CB}_{\text{RFNBO}}[\text{MJ}] = 0.02 \times \sum_{i}M_{i}$$",
                )
            ],
            id="a label beside a formula that is the whole right-hand side",
        ),
        pytest.param(
            one_row_table("Cap on additional energy =", "1.3 times the open water figure"),
            None,
            [(SectionKind.TABLE, "Cap on additional energy = | 1.3 times the open water figure")],
            id="a label with no formula beside it stays a table",
        ),
        pytest.param(
            '<table class="oj-table"><tbody>'
            '<tr class="oj-table"><td class="oj-table"><p>CB =</p></td>'
            f'<td class="oj-table"><p>{IMAGE}</p></td></tr>'
            '<tr class="oj-table"><td class="oj-table"><p>Where</p></td>'
            '<td class="oj-table"><p>the balance</p></td></tr>'
            "</tbody></table>",
            (RFNBO_EXPRESSION,),
            [(SectionKind.TABLE, f"CB = | {RFNBO_EXPRESSION}\nWhere | the balance")],
            id="a table of several rows stays a table",
        ),
    ],
)
def test_a_one_row_equation_table_is_read_as_one_formula(
    body: str, formulas: tuple[str, ...] | None, children: list[tuple[SectionKind, str]]
) -> None:
    (annex,) = annexes(parse_eurlex_html(annex_with(body), formulas))
    assert [
        (section.kind, section.text or "\n".join(" | ".join(row) for row in section.rows))
        for section in all_sections(annex.children)
    ] == children
