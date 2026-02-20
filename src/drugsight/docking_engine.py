"""
Molecular Docking Engine — wraps AutoDock Vina for binding-affinity prediction.

Handles PDB/SDF/MOL to PDBQT conversion via Open Babel and runs Vina docking
either for a single ligand-receptor pair or as a batch screen across a drug
library.  All binary checks are lazy (happen at call time, never at import
time) so that tests and dry-run workflows never require the binaries to be
installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from drugsight.config import RESULTS_DIR
from drugsight.schemas import BATCH_DOCKING_COLUMNS, DockingResult

logger = logging.getLogger(__name__)


# ── Binary availability helpers ──────────────────────────────────────


def _require_binary(name: str) -> str:
    """Return the absolute path of *name* on PATH, or raise RuntimeError."""
    # Check PATH first, then common local install locations.
    location = shutil.which(name)
    if location is None:
        for extra_dir in (Path.home() / "bin", Path("/usr/local/bin"), Path("/opt/homebrew/bin")):
            candidate = extra_dir / name
            if candidate.is_file():
                location = str(candidate)
                break
    if location is None:
        raise RuntimeError(
            f"'{name}' not found on PATH. "
            f"Install it before running docking.\n"
            f"  macOS:  brew install {name}\n"
            f"  Ubuntu: sudo apt-get install {name}\n"
            f"  conda:  conda install -c conda-forge {name}"
        )
    return location


# ── Format conversion ────────────────────────────────────────────────


def prepare_receptor(pdb_path: Path) -> Path:
    """Convert a receptor PDB file to PDBQT format for docking.

    Uses Open Babel with rigid-molecule mode and Gasteiger partial charges.
    The output file is written alongside the input with a ``.pdbqt`` extension.

    Parameters
    ----------
    pdb_path:
        Path to the receptor ``.pdb`` file.

    Returns
    -------
    Path
        The generated ``.pdbqt`` file.

    Raises
    ------
    RuntimeError
        If ``obabel`` is not on PATH or the conversion fails.
    """
    obabel = _require_binary("obabel")
    pdb_path = Path(pdb_path)
    output_path = pdb_path.with_suffix(".pdbqt")

    cmd = [
        obabel,
        str(pdb_path),
        "-O", str(output_path),
        "-xr",
        "--partialcharge", "gasteiger",
    ]
    logger.info("Preparing receptor: %s -> %s", pdb_path.name, output_path.name)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.debug("obabel stdout: %s", result.stdout.strip())

    if not output_path.exists():
        raise RuntimeError(
            f"obabel ran without error but output file was not created: {output_path}"
        )
    return output_path


def prepare_ligand(mol_path: Path) -> Path:
    """Convert a ligand MOL or SDF file to PDBQT format for docking.

    Uses Open Babel with 3-D coordinate generation and Gasteiger charges.
    The output file is written alongside the input with a ``.pdbqt`` extension.

    Parameters
    ----------
    mol_path:
        Path to the ligand ``.mol`` or ``.sdf`` file.

    Returns
    -------
    Path
        The generated ``.pdbqt`` file.

    Raises
    ------
    RuntimeError
        If ``obabel`` is not on PATH or the conversion fails.
    """
    obabel = _require_binary("obabel")
    mol_path = Path(mol_path)

    if mol_path.suffix.lower() not in (".mol", ".sdf"):
        logger.warning(
            "Unexpected ligand extension '%s'; Open Babel may still handle it.",
            mol_path.suffix,
        )

    output_path = mol_path.with_suffix(".pdbqt")

    cmd = [
        obabel,
        str(mol_path),
        "-O", str(output_path),
        "--gen3d",
        "--partialcharge", "gasteiger",
    ]
    logger.info("Preparing ligand: %s -> %s", mol_path.name, output_path.name)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.debug("obabel stdout: %s", result.stdout.strip())

    if not output_path.exists():
        raise RuntimeError(
            f"obabel ran without error but output file was not created: {output_path}"
        )
    return output_path


# ── Binding-site heuristic ───────────────────────────────────────────


def find_binding_center(pdb_path: Path) -> tuple[float, float, float]:
    """Estimate a binding-site center as the geometric mean of CA atoms.

    Parses the PDB file line-by-line according to the PDB fixed-column
    format and averages the (x, y, z) coordinates of every C-alpha atom.

    Parameters
    ----------
    pdb_path:
        Path to the receptor ``.pdb`` file.

    Returns
    -------
    tuple[float, float, float]
        ``(center_x, center_y, center_z)`` in angstroms.

    Raises
    ------
    ValueError
        If no CA atoms are found in the file.
    """
    pdb_path = Path(pdb_path)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    with open(pdb_path, "r") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16]
            if atom_name.strip() != "CA":
                continue
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))

    if not xs:
        raise ValueError(f"No CA atoms found in {pdb_path}; cannot determine center.")

    center = (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )
    logger.info(
        "Binding center from %d CA atoms: (%.2f, %.2f, %.2f)",
        len(xs), *center,
    )
    return center


# ── Single docking run ───────────────────────────────────────────────


def dock_single(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
    exhaustiveness: int = 8,
) -> DockingResult:
    """Run AutoDock Vina for one receptor-ligand pair.

    Parameters
    ----------
    receptor_pdbqt:
        Path to the receptor ``.pdbqt`` file.
    ligand_pdbqt:
        Path to the ligand ``.pdbqt`` file.
    center:
        ``(x, y, z)`` center of the search box in angstroms.
    box_size:
        ``(size_x, size_y, size_z)`` dimensions of the search box.
    exhaustiveness:
        Vina exhaustiveness parameter (higher = slower but more thorough).

    Returns
    -------
    DockingResult
        Dict with ``affinity_kcal_mol``, ``output_file``, and ``log_file``.

    Raises
    ------
    RuntimeError
        If ``vina`` is not on PATH, the subprocess fails, or the log file
        cannot be parsed.
    """
    vina = _require_binary("vina")
    receptor_pdbqt = Path(receptor_pdbqt)
    ligand_pdbqt = Path(ligand_pdbqt)

    output_dir = RESULTS_DIR / "docking"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{receptor_pdbqt.stem}_{ligand_pdbqt.stem}"
    output_file = output_dir / f"{stem}_out.pdbqt"
    log_file = output_dir / f"{stem}_log.txt"

    cmd = [
        vina,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--out", str(output_file),
    ]

    logger.info("Docking %s against %s", ligand_pdbqt.name, receptor_pdbqt.name)
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Vina 1.2.7+ prints the results table to stdout (no --log flag).
    # Save stdout as the log file for reference.
    log_file.write_text(proc.stdout)

    # Parse the results table from stdout.
    affinity = _parse_vina_output(proc.stdout)

    logger.info("Best affinity: %.2f kcal/mol (%s)", affinity, ligand_pdbqt.name)

    return DockingResult(
        affinity_kcal_mol=affinity,
        output_file=str(output_file),
        log_file=str(log_file),
    )


def _parse_vina_output(output: str) -> float:
    """Extract the top-ranked binding affinity from Vina's stdout.

    Vina output contains a results table whose first data row begins with
    ``   1``.  The second whitespace-separated token on that line is the
    affinity in kcal/mol.

    Raises
    ------
    RuntimeError
        If the expected table row is not found.
    """
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("1"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue

    raise RuntimeError(
        f"Could not parse binding affinity from Vina output:\n{output[:500]}"
    )


# ── Batch screening ─────────────────────────────────────────────────


def batch_dock(
    receptor_pdb: Path,
    drug_library_df: "pd.DataFrame",
    target_uniprot_id: str,
    target_symbol: str,
    center: tuple[float, float, float] | None = None,
) -> "pd.DataFrame":
    """Screen an entire drug library against a single receptor.

    For each drug in *drug_library_df* the ligand conformer is prepared,
    docked, and the result is collected.  Individual failures are logged
    and skipped — the batch never aborts early.

    Parameters
    ----------
    receptor_pdb:
        Path to the receptor ``.pdb`` file (will be converted to PDBQT).
    drug_library_df:
        DataFrame with at least ``drugbank_id``, ``name``, and
        ``conformer_path`` columns.
    target_uniprot_id:
        UniProt accession of the receptor.
    target_symbol:
        HGNC gene symbol of the receptor.
    center:
        Search-box center.  If ``None``, computed automatically via
        :func:`find_binding_center`.

    Returns
    -------
    pd.DataFrame
        Columns matching :data:`~drugsight.schemas.BATCH_DOCKING_COLUMNS`,
        sorted by ``affinity_kcal_mol`` ascending (best binders first).
    """
    import pandas as pd

    receptor_pdb = Path(receptor_pdb)

    # Prepare receptor once for the whole batch.
    receptor_pdbqt = prepare_receptor(receptor_pdb)

    if center is None:
        center = find_binding_center(receptor_pdb)

    total = len(drug_library_df)
    results: list[dict] = []

    for idx, row in drug_library_df.iterrows():
        drugbank_id: str = row["drugbank_id"]
        drug_name: str = row["name"]
        conformer_path = Path(row["conformer_path"])

        logger.info(
            "[%d/%d] Docking %s (%s)",
            len(results) + 1, total, drug_name, drugbank_id,
        )

        try:
            ligand_pdbqt = prepare_ligand(conformer_path)
            docking = dock_single(receptor_pdbqt, ligand_pdbqt, center)
            results.append(
                {
                    "drugbank_id": drugbank_id,
                    "drug_name": drug_name,
                    "uniprot_id": target_uniprot_id,
                    "target_symbol": target_symbol,
                    "affinity_kcal_mol": docking["affinity_kcal_mol"],
                    "output_file": docking["output_file"],
                    "log_file": docking["log_file"],
                }
            )
        except Exception:
            logger.exception(
                "Docking failed for %s (%s); skipping.", drug_name, drugbank_id,
            )

    df = pd.DataFrame(results, columns=BATCH_DOCKING_COLUMNS)
    df = df.sort_values("affinity_kcal_mol", ascending=True).reset_index(drop=True)

    logger.info(
        "Batch complete: %d/%d drugs docked successfully against %s.",
        len(df), total, target_symbol,
    )
    return df
