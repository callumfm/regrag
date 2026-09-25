"""THETIS-MRV entity lookup: every company and ship a name in the question could mean."""

import re

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import make_transient

from app.mrv.entities import find_entities
from tests.mrv.conftest import company_report

pytestmark = pytest.mark.anyio


FLEET = [
    company_report("SCORPIO KAIUN LTD.", "6349424"),
    company_report("SCORPIO KAIUN LTD.", "6349424", period=2025),
    company_report("SCORPIO CARRIERS, ltd.", "1575593"),
    company_report("Scorpio Marine Management (India) Private Limited", "5562457", period=2025),
    company_report("DFDS A/S", "0310102"),
    company_report("DFDS DENIZCILIK VE TASIMACILIK A.S.", "1568628"),
    company_report("Carras (Hellas) S.A.", "5123456"),
    company_report("Hellas Confidence Shipmanagement S.A.", "5999999"),
    company_report("Maersk A/S", "1234567", imo="9321483", ship_name="EMMA MAERSK"),
]


@pytest.fixture
async def fleet(db_session: AsyncSession) -> AsyncSession:
    for row in FLEET:
        make_transient(row)
    db_session.add_all(FLEET)
    await db_session.flush()
    return db_session


@pytest.mark.parametrize(
    ("question", "candidates", "years"),
    [
        pytest.param(
            "check the exposure of scorpio",
            [["6349424", "1575593", "5562457"]],
            "reports for 2025)",
            id="a lower-case name brings every company it starts, most reports first",
        ),
        pytest.param(
            "What was DFDS A/S's exposure in 2024",
            [["0310102"]],
            "reports for 2024)",
            id="a name given in full with its corporate ending settles the company",
        ),
        pytest.param(
            "What was DFDS's exposure in 2024",
            [["0310102", "1568628"]],
            None,
            id="a bare name brings its full match and the companies it starts",
        ),
        pytest.param(
            "What was the Carras Hellas exposure?",
            [["5123456"]],
            None,
            id="a word inside a name matched in full is not looked up again",
        ),
        pytest.param(
            "How much did Emma Maersk emit?",
            [["9321483"]],
            None,
            id="a ship's name keeps its company's word from being looked up",
        ),
        pytest.param(
            "SCORPIO KAIUN LTD. (IMO company number 6349424)",
            [["6349424"]],
            "reports for 2024 and 2025)",
            id="an option sent back resolves to its one company",
        ),
    ],
)
async def test_each_name_lists_every_candidate_it_could_mean(fleet, question, candidates, years):
    lines = await find_entities(fleet, question)

    assert [re.findall(r"\b\d{7}\b", line) for line in lines] == candidates
    if years:
        assert years in lines[0]
