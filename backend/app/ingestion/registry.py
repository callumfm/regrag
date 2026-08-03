"""Curated corpus registry: desired state, changed by PR. Fetch state lives in the DB."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSpec:
    """One document that belongs in the corpus; ref is the permanent source-specific id."""

    name: str
    source: str
    ref: str
    note: str = ""


@dataclass(frozen=True)
class Resolution:
    """A resolver's answer: a ref pinned to a version and a guaranteed-fetchable HTML URL."""

    resolved_ref: str
    url: str


# Pending acts watchlist (not yet adopted/published, revisit): FuelEU OPS communication;
# zero-emission tech acceptance criteria; on-board carbon capture; Annex II default values
# update; periodic transhipment-port list updates; COM(2026) 620 MRV/ETS/FuelEU
# restructuring proposal.
CORPUS: tuple[DocumentSpec, ...] = (
    DocumentSpec(
        name="fueleu-maritime",
        source="eurlex",
        ref="32023R1805",
        note="Regulation (EU) 2023/1805 (FuelEU Maritime); no consolidated HTML exists",
    ),
    DocumentSpec(
        name="ets-directive",
        source="eurlex",
        ref="32003L0087",
        note="Directive 2003/87/EC (EU ETS)",
    ),
    DocumentSpec(
        name="mrv-regulation",
        source="eurlex",
        ref="32015R0757",
        note="Regulation (EU) 2015/757 (MRV shipping)",
    ),
    DocumentSpec(
        name="fueleu-verification",
        source="eurlex",
        ref="32024R2027",
        note="IR (EU) 2024/2027 (FuelEU verification); consolidated HTML 404s",
    ),
    DocumentSpec(
        name="fueleu-monitoring-plan",
        source="eurlex",
        ref="32024R2031",
        note="IR (EU) 2024/2031 (FuelEU monitoring-plan template)",
    ),
    DocumentSpec(
        name="fueleu-verifier-accreditation",
        source="eurlex",
        ref="32025R0192",
        note="DR (EU) 2025/192 (FuelEU verifier accreditation)",
    ),
    DocumentSpec(
        name="fueleu-transhipment-ports",
        source="eurlex",
        ref="32025R1127",
        note="IR (EU) 2025/1127 (FuelEU transhipment ports)",
    ),
    DocumentSpec(
        name="fueleu-database",
        source="eurlex",
        ref="32026R0394",
        note="IR (EU) 2026/394 (FuelEU database)",
    ),
    DocumentSpec(
        name="ets-company-administration",
        source="eurlex",
        ref="32023R2599",
        note="IR (EU) 2023/2599 (ETS shipping-company administration)",
    ),
    DocumentSpec(
        name="ets-administering-authorities",
        source="eurlex",
        ref="32024D0411",
        note="ID (EU) 2024/411 (ETS administering-authority list)",
    ),
    DocumentSpec(
        name="ets-administering-authorities-correction",
        source="eurlex",
        ref="32026D1453",
        note="ID (EU) 2026/1453; supersedes 2024/411 annex from 2026-07-30",
    ),
    DocumentSpec(
        name="ets-transhipment-ports",
        source="eurlex",
        ref="32023R2297",
        note="IR (EU) 2023/2297 (ETS transhipment ports)",
    ),
    DocumentSpec(
        name="ets-derogation-lists",
        source="eurlex",
        ref="32023D2895",
        note="ID (EU) 2023/2895 (ETS island/PSO derogation lists)",
    ),
    DocumentSpec(
        name="mrv-templates",
        source="eurlex",
        ref="32023R2449",
        note="IR (EU) 2023/2449 (MRV templates)",
    ),
    DocumentSpec(
        name="mrv-verification",
        source="eurlex",
        ref="32023R2917",
        note="DR (EU) 2023/2917 (MRV verification + accreditation); consolidated HTML 404s",
    ),
    DocumentSpec(
        name="mrv-company-emissions",
        source="eurlex",
        ref="32023R2849",
        note="DR (EU) 2023/2849 (company-level aggregated emissions)",
    ),
    DocumentSpec(
        name="mrv-cargo-determination",
        source="eurlex",
        ref="32016R1928",
        note="IR (EU) 2016/1928 (MRV cargo-carried determination); consolidated HTML 404s",
    ),
)
