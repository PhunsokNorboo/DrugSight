"""
Tests for Module 2 — AlphaFold Structure Retrieval.

Validates that fetch_structure(), get_confidence(), and
fetch_structures_batch() conform to the AlphaFoldConfidence schema
contract.  All HTTP calls are mocked with the ``responses`` library.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import responses

from drugsight.config import ALPHAFOLD_API_URL
from drugsight.schemas import AlphaFoldConfidence

# Reset the module-level cached session before each test so that the
# ``responses`` mock transport is picked up.
import drugsight.alphafold_client as _af_mod

PREDICTION_URL = f"{ALPHAFOLD_API_URL}/api/prediction"
PDB_URL = f"{ALPHAFOLD_API_URL}/files"


@pytest.fixture(autouse=True)
def _reset_session():
    """Clear the module-level cached session between tests."""
    _af_mod._session = None
    yield
    _af_mod._session = None


# ── Helpers ──────────────────────────────────────────────────────────────

SAMPLE_PDB_CONTENT = (
    b"ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 90.00           N\n"
    b"ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 92.00           C\n"
    b"END\n"
)


def _prediction_json(
    uniprot_id: str,
    avg_plddt: float,
    version: int = 4,
) -> list[dict]:
    """Build a mock AlphaFold prediction API response."""
    return [
        {
            "uniprotAccession": uniprot_id,
            "confidenceAvgLocalScore": avg_plddt,
            "pdbUrl": f"{PDB_URL}/AF-{uniprot_id}-F1-model_v{version}.pdb",
            "latestVersion": version,
        }
    ]


# ── Tests ────────────────────────────────────────────────────────────────


@responses.activate
def test_fetch_structure_downloads_pdb(tmp_path: Path):
    """A 200 response writes a PDB file to the output directory."""
    from drugsight.alphafold_client import fetch_structure

    uid = "P04637"
    filename = f"AF-{uid}-F1-model_v4.pdb"
    responses.add(
        responses.GET,
        f"{PDB_URL}/{filename}",
        body=SAMPLE_PDB_CONTENT,
        status=200,
        content_type="chemical/x-pdb",
    )

    result = fetch_structure(uid, output_dir=tmp_path)

    assert result is not None
    assert result.exists()
    assert result.name == f"AF-{uid}-F1-model.pdb"
    assert result.read_bytes() == SAMPLE_PDB_CONTENT


@responses.activate
def test_fetch_structure_404_returns_none(tmp_path: Path):
    """A 404 from AlphaFold returns None (does not raise)."""
    from drugsight.alphafold_client import fetch_structure

    uid = "XNOPE"
    filename = f"AF-{uid}-F1-model_v4.pdb"
    responses.add(
        responses.GET,
        f"{PDB_URL}/{filename}",
        status=404,
    )

    result = fetch_structure(uid, output_dir=tmp_path)
    assert result is None


@responses.activate
def test_get_confidence_parses_json():
    """get_confidence returns an AlphaFoldConfidence dict with correct keys and types."""
    from drugsight.alphafold_client import get_confidence

    uid = "P42858"
    responses.add(
        responses.GET,
        f"{PREDICTION_URL}/{uid}",
        json=_prediction_json(uid, avg_plddt=85.3, version=4),
        status=200,
    )

    conf = get_confidence(uid)

    assert conf is not None

    # Verify all keys from AlphaFoldConfidence are present.
    expected_keys = set(AlphaFoldConfidence.__annotations__)
    assert set(conf.keys()) == expected_keys

    # Verify types.
    assert isinstance(conf["uniprot_id"], str)
    assert isinstance(conf["avg_plddt"], float)
    assert isinstance(conf["model_url"], str)
    assert isinstance(conf["version"], int)

    # Verify values.
    assert conf["uniprot_id"] == uid
    assert conf["avg_plddt"] == pytest.approx(85.3)
    assert conf["version"] == 4


@responses.activate
def test_fetch_structures_batch_filters_low_plddt(tmp_path: Path):
    """Batch fetch with min_plddt=70 keeps only the protein above threshold."""
    from drugsight.alphafold_client import fetch_structures_batch

    uid_low = "Q00001"  # pLDDT 50 — below threshold
    uid_high = "Q00002"  # pLDDT 80 — above threshold

    # Confidence endpoint for uid_low.
    responses.add(
        responses.GET,
        f"{PREDICTION_URL}/{uid_low}",
        json=_prediction_json(uid_low, avg_plddt=50.0),
        status=200,
    )

    # Confidence endpoint for uid_high.
    responses.add(
        responses.GET,
        f"{PREDICTION_URL}/{uid_high}",
        json=_prediction_json(uid_high, avg_plddt=80.0),
        status=200,
    )

    # PDB download for uid_high (uid_low should never reach this stage).
    filename_high = f"AF-{uid_high}-F1-model_v4.pdb"
    responses.add(
        responses.GET,
        f"{PDB_URL}/{filename_high}",
        body=SAMPLE_PDB_CONTENT,
        status=200,
        content_type="chemical/x-pdb",
    )

    paths, confidences = fetch_structures_batch(
        [uid_low, uid_high],
        output_dir=tmp_path,
        min_plddt=70.0,
    )

    assert len(paths) == 1
    assert len(confidences) == 1
    assert confidences[0]["uniprot_id"] == uid_high
    assert confidences[0]["avg_plddt"] >= 70.0


@responses.activate
def test_fetch_structure_caches(tmp_path: Path):
    """Calling fetch_structure twice for the same protein only downloads once."""
    from drugsight.alphafold_client import fetch_structure

    uid = "P12345"
    filename = f"AF-{uid}-F1-model_v4.pdb"
    responses.add(
        responses.GET,
        f"{PDB_URL}/{filename}",
        body=SAMPLE_PDB_CONTENT,
        status=200,
        content_type="chemical/x-pdb",
    )

    # First call — should hit the network.
    first = fetch_structure(uid, output_dir=tmp_path)
    assert first is not None

    # Second call — file already on disk, should NOT make another request.
    second = fetch_structure(uid, output_dir=tmp_path)
    assert second is not None
    assert first == second

    # Only one HTTP request should have been made.
    assert len(responses.calls) == 1
