"""
DrugSight shared data contracts.

Every module imports its input/output shapes from here.
This is the single source of truth — do not redefine these elsewhere.
"""

from __future__ import annotations

from typing import TypedDict


# ── Module 1 output ──────────────────────────────────────────────────
class TargetRecord(TypedDict):
    ensembl_id: str
    symbol: str
    name: str
    association_score: float
    uniprot_id: str


TARGET_COLUMNS = [
    "ensembl_id", "symbol", "name", "association_score", "uniprot_id",
]


# ── Module 2 output ──────────────────────────────────────────────────
class AlphaFoldConfidence(TypedDict):
    uniprot_id: str
    avg_plddt: float
    model_url: str
    version: int


# ── Module 3 output ──────────────────────────────────────────────────
DRUG_LIBRARY_COLUMNS = [
    "drugbank_id", "name", "smiles", "mol_weight", "logp",
    "hbd", "hba", "tpsa", "rotatable_bonds",
]


# ── Module 4 output ──────────────────────────────────────────────────
class DockingResult(TypedDict):
    affinity_kcal_mol: float
    output_file: str
    log_file: str


BATCH_DOCKING_COLUMNS = [
    "drugbank_id", "drug_name", "uniprot_id", "target_symbol",
    "affinity_kcal_mol", "output_file", "log_file",
]


# ── Module 5 features & output ───────────────────────────────────────
SCORING_FEATURES = [
    "binding_affinity",
    "plddt_score",
    "bioactivity_count",
    "safety_score",
    "tanimoto_similarity",
    "association_score",
    "patent_expired",
    "oral_bioavailability",
]

RANKED_RESULT_COLUMNS = (
    BATCH_DOCKING_COLUMNS
    + SCORING_FEATURES
    + ["composite_score", "top_contributing_factor"]
)
