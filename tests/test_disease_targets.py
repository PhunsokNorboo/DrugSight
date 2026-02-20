"""
Tests for Module 1 — Disease-to-Target Mapping.

Validates that get_disease_targets() conforms to the TargetRecord /
TARGET_COLUMNS schema contract.  All HTTP interactions (Open Targets
GraphQL API, UniProt ID Mapping) are mocked using the ``responses``
library so tests never hit the network.
"""

from __future__ import annotations

import pytest
import responses

from drugsight.config import OPEN_TARGETS_URL, UNIPROT_API_URL
from drugsight.disease_targets import get_disease_targets
from drugsight.schemas import TARGET_COLUMNS


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_open_targets_body(
    disease_name: str,
    rows: list[dict],
) -> dict:
    """Build a realistic Open Targets GraphQL response body."""
    return {
        "data": {
            "disease": {
                "name": disease_name,
                "associatedTargets": {
                    "rows": rows,
                },
            },
        },
    }


def _target_row(ensembl_id: str, symbol: str, name: str, score: float) -> dict:
    """Shorthand for a single Open Targets target row."""
    return {
        "target": {
            "id": ensembl_id,
            "approvedSymbol": symbol,
            "approvedName": name,
        },
        "score": score,
    }


def _register_uniprot_mocks(
    mapping: dict[str, str],
) -> None:
    """Register UniProt submit + poll + results mocks for *mapping*.

    ``mapping`` maps Ensembl IDs to UniProt accessions.
    """
    # Step 1 — Submit job.
    responses.add(
        responses.POST,
        f"{UNIPROT_API_URL}/idmapping/run",
        json={"jobId": "test-job-123"},
        status=200,
    )

    # Step 2 — Poll returns FINISHED.
    responses.add(
        responses.GET,
        f"{UNIPROT_API_URL}/idmapping/status/test-job-123",
        json={"jobStatus": "FINISHED"},
        status=200,
    )

    # Step 3 — Fetch results.
    results = [
        {"from": eid, "to": {"primaryAccession": acc}}
        for eid, acc in mapping.items()
    ]
    responses.add(
        responses.GET,
        f"{UNIPROT_API_URL}/idmapping/uniprotkb/results/test-job-123",
        json={"results": results},
        status=200,
    )


# ── Tests ────────────────────────────────────────────────────────────────


@responses.activate
def test_get_disease_targets_returns_dataframe():
    """Mocked API call returns a DataFrame whose columns match TARGET_COLUMNS."""
    rows = [
        _target_row("ENSG00000197386", "HTT", "Huntingtin", 0.95),
        _target_row("ENSG00000176697", "BDNF", "Brain-derived neurotrophic factor", 0.82),
    ]
    responses.add(
        responses.POST,
        OPEN_TARGETS_URL,
        json=_make_open_targets_body("Huntington disease", rows),
        status=200,
    )
    _register_uniprot_mocks({
        "ENSG00000197386": "P42858",
        "ENSG00000176697": "P23560",
    })

    df = get_disease_targets("EFO_0000337", min_score=0.5)

    assert list(df.columns) == TARGET_COLUMNS
    assert len(df) == 2
    # Every column should be non-null for these mocked records.
    assert df["ensembl_id"].notna().all()
    assert df["symbol"].notna().all()
    assert df["name"].notna().all()
    assert df["association_score"].notna().all()
    assert df["uniprot_id"].notna().all()


@responses.activate
def test_get_disease_targets_filters_by_score():
    """With min_score=0.5, only targets scoring >= 0.5 are returned."""
    rows = [
        _target_row("ENSG00000000001", "GENE_A", "Gene A", 0.3),
        _target_row("ENSG00000000002", "GENE_B", "Gene B", 0.7),
        _target_row("ENSG00000000003", "GENE_C", "Gene C", 0.9),
    ]
    responses.add(
        responses.POST,
        OPEN_TARGETS_URL,
        json=_make_open_targets_body("Test disease", rows),
        status=200,
    )
    _register_uniprot_mocks({
        "ENSG00000000002": "Q00002",
        "ENSG00000000003": "Q00003",
    })

    df = get_disease_targets("EFO_0009999", min_score=0.5)

    assert len(df) == 2
    # The low-scoring target must not appear.
    assert "GENE_A" not in df["symbol"].values
    assert set(df["symbol"].values) == {"GENE_B", "GENE_C"}
    # All scores must be at or above the threshold.
    assert (df["association_score"] >= 0.5).all()


@responses.activate
def test_get_disease_targets_empty_response():
    """An empty result set from Open Targets raises ValueError."""
    body = {
        "data": {
            "disease": {
                "name": "Unknown disease",
                "associatedTargets": {"rows": []},
            }
        }
    }
    responses.add(
        responses.POST,
        OPEN_TARGETS_URL,
        json=body,
        status=200,
    )

    with pytest.raises(ValueError, match="No targets found"):
        get_disease_targets("EFO_0000000", min_score=0.1)


@responses.activate
def test_get_disease_targets_sorted_descending():
    """The returned DataFrame is sorted by association_score in descending order."""
    rows = [
        _target_row("ENSG00000000010", "LOW", "Low scorer", 0.55),
        _target_row("ENSG00000000011", "MID", "Mid scorer", 0.75),
        _target_row("ENSG00000000012", "HIGH", "High scorer", 0.92),
    ]
    responses.add(
        responses.POST,
        OPEN_TARGETS_URL,
        json=_make_open_targets_body("Sort test disease", rows),
        status=200,
    )
    _register_uniprot_mocks({
        "ENSG00000000010": "Q00010",
        "ENSG00000000011": "Q00011",
        "ENSG00000000012": "Q00012",
    })

    df = get_disease_targets("EFO_0000555", min_score=0.5)

    scores = df["association_score"].tolist()
    assert scores == sorted(scores, reverse=True), (
        f"Expected descending sort but got: {scores}"
    )
    assert df.iloc[0]["symbol"] == "HIGH"
    assert df.iloc[-1]["symbol"] == "LOW"
