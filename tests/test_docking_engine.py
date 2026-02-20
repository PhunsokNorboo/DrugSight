"""
Tests for Module 4 — Molecular Docking Engine.

Validates that find_binding_center(), dock_single(), batch_dock(),
prepare_receptor(), and _require_binary() conform to the DockingResult /
BATCH_DOCKING_COLUMNS schema contracts.

AutoDock Vina and Open Babel subprocess calls are mocked so tests run
without external binaries.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from drugsight.schemas import BATCH_DOCKING_COLUMNS, DockingResult


# ── Helpers ──────────────────────────────────────────────────────────────

# Minimal PDB content with 3 CA atoms at known coordinates.
_PDB_3CA = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 90.00           N
ATOM      2  CA  ALA A   1      10.000  20.000  30.000  1.00 92.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 88.00           C
ATOM      4  N   GLY A   2       4.000   5.000   6.000  1.00 85.00           N
ATOM      5  CA  GLY A   2      40.000  50.000  60.000  1.00 87.00           C
ATOM      6  C   GLY A   2       6.000   7.000   8.000  1.00 86.00           C
ATOM      7  N   VAL A   3       7.000   8.000   9.000  1.00 83.00           N
ATOM      8  CA  VAL A   3      70.000  80.000  90.000  1.00 91.00           C
ATOM      9  C   VAL A   3       9.000  10.000  11.000  1.00 89.00           C
END
"""

# Sample Vina log content with a results table.
_VINA_LOG = """\
AutoDock Vina v1.2.5
Detected 4 CPUs.
mode |   affinity | dist from best mode
-----+------------+--------------------
   1         -7.8              0.000
   2         -7.2              1.543
   3         -6.9              2.108
"""


# ── Tests ────────────────────────────────────────────────────────────────


def test_find_binding_center_returns_tuple(tmp_path: Path):
    """Parsing a PDB with 3 CA atoms returns a 3-tuple of floats (geometric mean)."""
    from drugsight.docking_engine import find_binding_center

    pdb_file = tmp_path / "test_receptor.pdb"
    pdb_file.write_text(_PDB_3CA)

    center = find_binding_center(pdb_file)

    assert isinstance(center, tuple)
    assert len(center) == 3
    assert all(isinstance(v, float) for v in center)

    # Expected: mean of (10, 40, 70), (20, 50, 80), (30, 60, 90)
    expected_x = (10.0 + 40.0 + 70.0) / 3
    expected_y = (20.0 + 50.0 + 80.0) / 3
    expected_z = (30.0 + 60.0 + 90.0) / 3
    assert center[0] == pytest.approx(expected_x)
    assert center[1] == pytest.approx(expected_y)
    assert center[2] == pytest.approx(expected_z)


@patch("drugsight.docking_engine.shutil.which", return_value="/usr/local/bin/vina")
@patch("drugsight.docking_engine.subprocess.run")
def test_dock_single_parses_log(
    mock_run: MagicMock,
    mock_which: MagicMock,
    tmp_path: Path,
):
    """dock_single parses the Vina log and returns a DockingResult with correct keys."""
    from drugsight.docking_engine import dock_single

    # Create dummy input files.
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("DUMMY RECEPTOR")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text("DUMMY LIGAND")

    # Make subprocess.run succeed and create the mock output file.
    def side_effect(cmd, **kwargs):
        # Parse the --out argument from the command.
        out_idx = cmd.index("--out")
        out_file = Path(cmd[out_idx + 1])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("DOCKED OUTPUT")
        return MagicMock(returncode=0, stdout=_VINA_LOG, stderr="")

    mock_run.side_effect = side_effect

    result = dock_single(
        receptor,
        ligand,
        center=(40.0, 50.0, 60.0),
    )

    # Verify DockingResult keys.
    expected_keys = set(DockingResult.__annotations__)
    assert set(result.keys()) == expected_keys

    # Verify types.
    assert isinstance(result["affinity_kcal_mol"], float)
    assert isinstance(result["output_file"], str)
    assert isinstance(result["log_file"], str)

    # Verify the parsed affinity from the mock log.
    assert result["affinity_kcal_mol"] == pytest.approx(-7.8)


@patch("drugsight.docking_engine.subprocess.run")
@patch("drugsight.docking_engine.shutil.which")
def test_batch_dock_handles_failures(
    mock_which: MagicMock,
    mock_run: MagicMock,
    tmp_path: Path,
):
    """One success and one failure in batch_dock returns a partial DataFrame."""
    import subprocess as _subprocess

    from drugsight.docking_engine import batch_dock

    # Return the correct binary path depending on the name requested.
    def which_side_effect(name):
        return {
            "obabel": "/usr/local/bin/obabel",
            "vina": "/usr/local/bin/vina",
        }.get(name)

    mock_which.side_effect = which_side_effect

    # Create a dummy receptor PDB (with CA atoms for center detection).
    receptor_pdb = tmp_path / "receptor.pdb"
    receptor_pdb.write_text(_PDB_3CA)

    # Create the PDBQT that prepare_receptor will "produce".
    receptor_pdbqt = receptor_pdb.with_suffix(".pdbqt")

    # Drug library with two drugs.
    conformer_a = tmp_path / "DB00001.mol"
    conformer_b = tmp_path / "DB00002.mol"
    conformer_a.write_text("DRUG A MOL")
    conformer_b.write_text("DRUG B MOL")

    drug_df = pd.DataFrame({
        "drugbank_id": ["DB00001", "DB00002"],
        "name": ["DrugA", "DrugB"],
        "conformer_path": [str(conformer_a), str(conformer_b)],
    })

    def side_effect(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)

        # prepare_receptor (obabel for receptor PDB -> PDBQT) — succeed.
        if "/obabel" in cmd_str and str(receptor_pdb) in cmd_str:
            receptor_pdbqt.write_text("RECEPTOR PDBQT")
            return MagicMock(returncode=0, stdout="", stderr="")

        # prepare_ligand for DrugA (obabel for ligand MOL -> PDBQT) — succeed.
        if "/obabel" in cmd_str and "DB00001" in cmd_str:
            pdbqt = conformer_a.with_suffix(".pdbqt")
            pdbqt.write_text("LIGAND A PDBQT")
            return MagicMock(returncode=0, stdout="", stderr="")

        # dock_single for DrugA (vina) — succeed: create output file, return results in stdout.
        if "/vina" in cmd_str and "--out" in cmd_str:
            out_idx = cmd.index("--out")
            out_file = Path(cmd[out_idx + 1])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("DOCKED OUTPUT A")
            return MagicMock(returncode=0, stdout=_VINA_LOG, stderr="")

        # prepare_ligand for DrugB (obabel) — FAIL.
        if "/obabel" in cmd_str and "DB00002" in cmd_str:
            raise _subprocess.CalledProcessError(1, cmd, "obabel error")

        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = side_effect

    result_df = batch_dock(
        receptor_pdb=receptor_pdb,
        drug_library_df=drug_df,
        target_uniprot_id="P42858",
        target_symbol="HTT",
        center=(40.0, 50.0, 60.0),
    )

    # The batch should have at least partial results (DrugA succeeded).
    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == BATCH_DOCKING_COLUMNS

    # DrugA should be present, DrugB should have been skipped due to failure.
    assert len(result_df) == 1
    assert result_df.iloc[0]["drugbank_id"] == "DB00001"


def test_missing_vina_binary_raises_error():
    """When 'vina' is not found anywhere, _require_binary raises RuntimeError."""
    from drugsight.docking_engine import _require_binary

    with patch("drugsight.docking_engine.shutil.which", return_value=None):
        with patch("drugsight.docking_engine.Path.is_file", return_value=False):
            with pytest.raises(RuntimeError, match="not found on PATH"):
                _require_binary("vina")


@patch("drugsight.docking_engine.subprocess.run")
@patch("drugsight.docking_engine.shutil.which", return_value="/usr/local/bin/obabel")
def test_prepare_receptor_calls_obabel(
    mock_which: MagicMock,
    mock_run: MagicMock,
    tmp_path: Path,
):
    """prepare_receptor invokes obabel with correct arguments for PDB -> PDBQT."""
    from drugsight.docking_engine import prepare_receptor

    pdb_file = tmp_path / "protein.pdb"
    pdb_file.write_text(_PDB_3CA)
    expected_out = pdb_file.with_suffix(".pdbqt")

    def side_effect(cmd, **kwargs):
        # Create the output file so the post-run check passes.
        expected_out.write_text("CONVERTED PDBQT")
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = side_effect

    result = prepare_receptor(pdb_file)

    assert result == expected_out
    assert expected_out.exists()

    # Verify subprocess was called exactly once.
    mock_run.assert_called_once()

    # Inspect the command that was passed.
    call_args = mock_run.call_args
    cmd = call_args[0][0]  # positional arg 0 is the command list

    assert cmd[0] == "/usr/local/bin/obabel"
    assert str(pdb_file) in cmd
    assert "-O" in cmd
    assert str(expected_out) in cmd
    assert "-xr" in cmd
    assert "--partialcharge" in cmd
    assert "gasteiger" in cmd
