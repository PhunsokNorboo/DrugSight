"""
Tests for Module 3 — Drug Library Preparation.

Validates that compute_descriptors(), load_drug_library(),
generate_conformer(), and prepare_library() conform to the
DRUG_LIBRARY_COLUMNS schema contract.

No HTTP mocking is needed — all operations are local (RDKit, CSV).
RDKit-dependent tests are conditionally skipped when RDKit is not
installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drugsight.schemas import DRUG_LIBRARY_COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── RDKit availability guard ─────────────────────────────────────────────


def _has_rdkit() -> bool:
    """Return True if the RDKit package is importable."""
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        return False


requires_rdkit = pytest.mark.skipif(
    not _has_rdkit(),
    reason="RDKit not installed",
)


# ── Tests ────────────────────────────────────────────────────────────────


@requires_rdkit
def test_compute_descriptors_valid_smiles():
    """Aspirin SMILES yields a dict with all six descriptor keys."""
    from drugsight.drug_library import compute_descriptors

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    desc = compute_descriptors(aspirin_smiles)

    assert desc is not None

    expected_keys = {"mol_weight", "logp", "hbd", "hba", "tpsa", "rotatable_bonds"}
    assert set(desc.keys()) == expected_keys

    # Type checks.
    assert isinstance(desc["mol_weight"], float)
    assert isinstance(desc["logp"], float)
    assert isinstance(desc["hbd"], int)
    assert isinstance(desc["hba"], int)
    assert isinstance(desc["tpsa"], float)
    assert isinstance(desc["rotatable_bonds"], int)

    # Sanity checks for aspirin (MW ~180).
    assert 170.0 < desc["mol_weight"] < 190.0
    assert desc["hbd"] >= 1
    assert desc["hba"] >= 3


@requires_rdkit
def test_compute_descriptors_invalid_smiles():
    """An invalid SMILES string returns None (not an exception)."""
    from drugsight.drug_library import compute_descriptors

    result = compute_descriptors("INVALID")
    assert result is None


@requires_rdkit
def test_load_drug_library_columns():
    """Loading sample_drugbank.csv produces a DataFrame with DRUG_LIBRARY_COLUMNS."""
    from drugsight.drug_library import load_drug_library

    csv_path = DATA_DIR / "sample_drugbank.csv"
    if not csv_path.exists():
        pytest.skip(f"Sample file not found: {csv_path}")

    df = load_drug_library(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == DRUG_LIBRARY_COLUMNS
    assert len(df) > 0

    # All approved rows should have valid SMILES (no nulls after descriptor computation).
    assert df["smiles"].notna().all()
    assert df["drugbank_id"].notna().all()
    assert df["mol_weight"].notna().all()


@requires_rdkit
def test_generate_conformer_creates_file(tmp_path: Path):
    """Generating a conformer for aspirin creates a MOL file on disk."""
    from drugsight.drug_library import generate_conformer

    aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    output = tmp_path / "aspirin.mol"

    result = generate_conformer(aspirin_smiles, output)

    assert result is not None
    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


@requires_rdkit
def test_prepare_library_adds_conformer_path(tmp_path: Path):
    """Full pipeline adds a 'conformer_path' column to the drug library DataFrame."""
    from drugsight.drug_library import prepare_library

    csv_path = DATA_DIR / "sample_drugbank.csv"
    if not csv_path.exists():
        pytest.skip(f"Sample file not found: {csv_path}")

    conformer_dir = tmp_path / "conformers"

    df = prepare_library(csv_path, conformer_dir=conformer_dir)

    # Must have all standard columns plus conformer_path.
    for col in DRUG_LIBRARY_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"
    assert "conformer_path" in df.columns

    # At least some conformers should have been successfully generated.
    generated = df["conformer_path"].notna().sum()
    assert generated > 0, "Expected at least one conformer to be generated"

    # Verify that the referenced files actually exist on disk.
    for path_str in df["conformer_path"].dropna():
        assert Path(path_str).exists(), f"Conformer file missing: {path_str}"
