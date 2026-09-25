"""THETIS-MRV entities: the dataset, and every company and ship a name in the question could
mean, found by stored key."""

import re
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from app.mrv.names import company_key, name_key, name_words
from app.mrv.schemas import MrvReport
from app.retrieval.search import words_in_corpus

MAX_NAME_WORDS = 5
ENTITY_LIMIT = 5
CANDIDATE_LIMIT = 5
"""The most candidates one name's line lists, most reports first; the line says how many in all."""
IMO_NUMBER = re.compile(r"\d{7}")


class NameRun(NamedTuple):
    """A run of consecutive question words that may name a company or ship, and where it sits."""

    start: int
    end: int
    words: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.words)

    def lies_inside(self, other: "NameRun") -> bool:
        return other != self and other.start <= self.start and self.end <= other.end


class Candidate(NamedTuple):
    """One company or ship a name could mean: its name, IMO number, stored key, the periods it
    reported in, and how many reports it filed."""

    name: str
    imo: str
    key: str
    periods: list[int]
    reports: int


class EntityKind(NamedTuple):
    """Companies or ships: how they are called, numbered, and stored."""

    noun: str
    plural: str
    number: str
    name: InstrumentedAttribute[Any]
    imo: InstrumentedAttribute[Any]
    key: InstrumentedAttribute[Any]
    to_key: Callable[[str], str]


COMPANY = EntityKind(
    "company",
    "companies",
    "IMO company number",
    MrvReport.company_name,
    MrvReport.company_imo,
    MrvReport.company_key,
    company_key,
)
SHIP = EntityKind(
    "ship", "ships", "IMO", MrvReport.ship_name, MrvReport.imo, MrvReport.ship_key, name_key
)


class NameMatch(NamedTuple):
    """The candidates one run of the question matched."""

    kind: EntityKind
    run: NameRun
    candidates: list[Candidate]


def word_runs(words: Sequence[str]) -> list[NameRun]:
    """Every run of up to MAX_NAME_WORDS consecutive words."""
    return [
        NameRun(start, end, tuple(words[start:end]))
        for start in range(len(words))
        for end in range(start + 1, min(start + MAX_NAME_WORDS, len(words)) + 1)
    ]


def describe_years(periods: Sequence[int]) -> str:
    """'2024 and 2025', or '2018 to 2025' for a longer unbroken span."""
    years = sorted(periods)
    if len(years) > 2 and years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]} to {years[-1]}"
    return " and ".join(str(year) for year in years)


def describe_candidate(kind: EntityKind, candidate: Candidate) -> str:
    return (
        f"{candidate.name} ({kind.number} {candidate.imo}, "
        f"reports for {describe_years(candidate.periods)})"
    )


def describe_match(match: NameMatch) -> str:
    """One candidate by name and number, or every candidate the name could mean."""
    kind, run, candidates = match
    if len(candidates) == 1:
        return f"{describe_candidate(kind, candidates[0])}, a {kind.noun} in THETIS-MRV"
    listed = "; ".join(describe_candidate(kind, c) for c in candidates[:CANDIDATE_LIMIT])
    most = (
        f", the {CANDIDATE_LIMIT} with most reports being"
        if len(candidates) > CANDIDATE_LIMIT
        else ""
    )
    return (
        f"'{run.text}', which could be any of {len(candidates)} {kind.plural} in "
        f"THETIS-MRV{most}: {listed}"
    )


def run_matches(kind: EntityKind, run: NameRun, candidate: Candidate, *, probed: bool) -> bool:
    """Whether the run names the candidate by number or in full, or, probed, starts its name."""
    return (
        candidate.imo == run.text
        or candidate.key == kind.to_key(run.text)
        or (probed and candidate.key.startswith(f"{name_key(run.text)} "))
    )


async def find_candidates(
    session: AsyncSession, kind: EntityKind, named: list[NameRun], probed: list[NameRun]
) -> list[NameMatch]:
    """Each run's candidates of one kind: those it names in full or by number, and for a probed
    run those whose name it starts."""
    if not named:
        return []
    starts = {name_key(run.text) for run in probed}
    stmt = (
        select(
            func.max(kind.name).label("name"),
            kind.imo.label("imo"),
            kind.key.label("key"),
            func.array_agg(MrvReport.period.distinct()).label("periods"),
            func.count().label("reports"),
        )
        .where(
            kind.imo.is_not(None),
            or_(
                kind.key.in_({kind.to_key(run.text) for run in named}),
                kind.imo.in_({run.text for run in named}),
                *(kind.key.startswith(f"{start} ") for start in starts),
            ),
        )
        .group_by(kind.key, kind.imo)
        .order_by(func.count().desc(), func.max(kind.name))
    )
    rows = [Candidate(*row) for row in await session.execute(stmt)]
    matches = []
    for run in named:
        probe = run in probed
        found = [row for row in rows if run_matches(kind, run, row, probed=probe)]
        if found:
            matches.append(NameMatch(kind, run, found))
    return matches


async def find_entities(session: AsyncSession, question: str) -> tuple[str, ...]:
    """The dataset itself, and for each name the question gives every company or ship it could
    mean, each with its IMO number and reporting years; a name inside a longer name that
    matched is not looked up on its own."""
    words = name_words(question)
    runs = word_runs(words)
    single = {word for word in words if len(word) >= 3}
    uncommon = single - await words_in_corpus(session, sorted(single))
    numbered = [run for run in runs if IMO_NUMBER.fullmatch(run.text)]
    company_runs = [
        run for run in runs if len(run.words) > 1 or run.words[0] in uncommon or run in numbered
    ]
    probed = [run for run in company_runs if run.words[0] in uncommon and run not in numbered]
    ship_runs = [run for run in runs if len(run.words) > 1] + numbered
    matches = [
        *await find_candidates(session, COMPANY, company_runs, probed),
        *await find_candidates(session, SHIP, ship_runs, []),
    ]
    kept = [
        match for match in matches if not any(match.run.lies_inside(other.run) for other in matches)
    ]
    dataset = ["THETIS-MRV, the dataset mrv_query reads"] if "thetis" in single else []
    lines = [*dataset, *(describe_match(match) for match in kept)]
    return tuple(dict.fromkeys(lines))[:ENTITY_LIMIT]
