"""
Drug Library Preparation (Module 3).

Load FDA-approved drugs from a CSV, compute molecular descriptors
using RDKit, and generate 3D conformers for molecular docking.

RDKit is imported lazily inside each function so this module remains
importable in environments where RDKit is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from drugsight.config import CONFORMERS_DIR
from drugsight.schemas import DRUG_LIBRARY_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_drug_library(drugbank_csv: str | Path) -> pd.DataFrame:
    """Load FDA-approved drugs from *drugbank_csv*.

    The CSV must contain at least ``drugbank_id``, ``name``, ``smiles``,
    and ``status`` columns.  Only rows where ``status == "approved"`` are
    kept.  Molecular descriptors are computed for every retained row and
    the result is returned with exactly :pydata:`DRUG_LIBRARY_COLUMNS`.

    Parameters
    ----------
    drugbank_csv:
        Path to the DrugBank CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns match ``DRUG_LIBRARY_COLUMNS``.
        Rows with unparseable SMILES are dropped.
    """
    csv_path = Path(drugbank_csv)
    logger.info("Loading drug library from %s", csv_path)

    df = pd.read_csv(csv_path)

    # Normalise column names so minor casing differences are tolerated.
    df.columns = df.columns.str.strip().str.lower()

    required = {"drugbank_id", "name", "smiles", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    # Keep only approved drugs.
    approved = df.loc[df["status"].str.lower() == "approved"].copy()
    logger.info(
        "Filtered to %d approved drugs (of %d total)", len(approved), len(df)
    )

    # Compute descriptors for each SMILES string.
    records: list[dict] = []
    for _, row in approved.iterrows():
        smiles = str(row["smiles"]).strip()
        desc = compute_descriptors(smiles)
        if desc is None:
            logger.warning(
                "Skipping %s (%s): unparseable SMILES '%s'",
                row["drugbank_id"],
                row["name"],
                smiles,
            )
            continue
        records.append(
            {
                "drugbank_id": row["drugbank_id"],
                "name": row["name"],
                "smiles": smiles,
                **desc,
            }
        )

    result = pd.DataFrame(records, columns=DRUG_LIBRARY_COLUMNS)

    # Use nullable integer types for columns that may contain NaN after
    # future joins or filters.
    int_cols = ["hbd", "hba", "rotatable_bonds"]
    for col in int_cols:
        result[col] = result[col].astype("Int64")

    logger.info("Drug library ready: %d drugs with descriptors", len(result))
    return result


def compute_descriptors(smiles: str) -> dict | None:
    """Compute Lipinski-relevant molecular descriptors for *smiles*.

    Parameters
    ----------
    smiles:
        A SMILES string representing the molecule.

    Returns
    -------
    dict | None
        Dictionary with keys ``mol_weight``, ``logp``, ``hbd``, ``hba``,
        ``tpsa``, ``rotatable_bonds``.  Returns ``None`` when RDKit
        cannot parse the SMILES.
    """
    from rdkit import Chem  # lazy import
    from rdkit.Chem import Descriptors  # lazy import

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "mol_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
    }


def generate_conformer(smiles: str, output_path: Path) -> Path | None:
    """Generate a 3D conformer and write it to *output_path* as a MOL file.

    If *output_path* already exists the file is **not** regenerated and the
    path is returned immediately (cache behaviour).

    Parameters
    ----------
    smiles:
        SMILES string of the molecule.
    output_path:
        Destination ``.mol`` file.

    Returns
    -------
    Path | None
        The written file path, or ``None`` if conformer generation failed.
    """
    output_path = Path(output_path)

    # Cache: skip if already generated.
    if output_path.exists():
        logger.debug("Conformer already cached: %s", output_path)
        return output_path

    from rdkit import Chem  # lazy import
    from rdkit.Chem import AllChem  # lazy import

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Cannot generate conformer — invalid SMILES: %s", smiles)
        return None

    try:
        mol = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if embed_status == -1:
            logger.warning(
                "EmbedMolecule failed for SMILES: %s", smiles
            )
            return None

        AllChem.MMFFOptimizeMolecule(mol)

        # Ensure the parent directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        Chem.MolToMolFile(mol, str(output_path))
        logger.debug("Conformer written: %s", output_path)
        return output_path

    except Exception:
        logger.exception("Conformer generation failed for SMILES: %s", smiles)
        return None


def prepare_library(
    drugbank_csv: str | Path,
    conformer_dir: Path | None = None,
) -> pd.DataFrame:
    """Full pipeline: load CSV, compute descriptors, generate conformers.

    Parameters
    ----------
    drugbank_csv:
        Path to the DrugBank CSV.
    conformer_dir:
        Directory for conformer ``.mol`` files.  Defaults to
        :pydata:`config.CONFORMERS_DIR`.

    Returns
    -------
    pd.DataFrame
        The drug library DataFrame (``DRUG_LIBRARY_COLUMNS`` plus a
        ``conformer_path`` column).
    """
    if conformer_dir is None:
        conformer_dir = CONFORMERS_DIR
    conformer_dir = Path(conformer_dir)
    conformer_dir.mkdir(parents=True, exist_ok=True)

    df = load_drug_library(drugbank_csv)

    conformer_paths: list[str | None] = []
    for _, row in df.iterrows():
        drug_id = str(row["drugbank_id"])
        smiles = str(row["smiles"])
        out_path = conformer_dir / f"{drug_id}.mol"

        result = generate_conformer(smiles, out_path)
        if result is not None:
            conformer_paths.append(str(result))
            logger.info(
                "Conformer ready for %s (%s)", drug_id, row["name"]
            )
        else:
            conformer_paths.append(None)
            logger.warning(
                "No conformer for %s (%s)", drug_id, row["name"]
            )

    df["conformer_path"] = conformer_paths
    logger.info(
        "Library prepared: %d drugs, %d conformers",
        len(df),
        df["conformer_path"].notna().sum(),
    )
    return df
