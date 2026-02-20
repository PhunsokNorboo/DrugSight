"""
Integration tests — module-to-module data flow in the DrugSight pipeline.

These tests verify that output from one module can be consumed by the next
module in the pipeline chain, with HTTP calls mocked via the ``responses``
library and filesystem operations routed through ``tmp_path``.

Module chain under test:
    disease_targets -> alphafold_client -> [drug_library -> docking_engine] -> ml_scorer -> pipeline

Individual module logic is covered in the per-module test files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import responses

from drugsight.config import ALPHAFOLD_API_URL, OPEN_TARGETS_URL, UNIPROT_API_URL
from drugsight.schemas import (
    BATCH_DOCKING_COLUMNS,
    SCORING_FEATURES,
    TARGET_COLUMNS,
    AlphaFoldConfidence,
)


# ---------------------------------------------------------------------------
# Shared helpers — realistic mock data builders
# ---------------------------------------------------------------------------

PREDICTION_URL = f"{ALPHAFOLD_API_URL}/api/prediction"
PDB_URL = f"{ALPHAFOLD_API_URL}/files"

SAMPLE_PDB_CONTENT = (
    b"ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 90.00           N\n"
    b"ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 92.00           C\n"
    b"END\n"
)


def _open_targets_body(rows: list[dict]) -> dict:
    """Build a realistic Open Targets GraphQL response body."""
    return {
        "data": {
            "disease": {
                "name": "Huntington disease",
                "associatedTargets": {"rows": rows},
            },
        },
    }


def _target_row(ensembl_id: str, symbol: str, name: str, score: float) -> dict:
    """Shorthand for a single Open Targets target row."""
    return {
        "target": {"id": ensembl_id, "approvedSymbol": symbol, "approvedName": name},
        "score": score,
    }


def _register_uniprot_mocks(mapping: dict[str, str]) -> None:
    """Register UniProt submit + poll + results mocks."""
    responses.add(
        responses.POST,
        f"{UNIPROT_API_URL}/idmapping/run",
        json={"jobId": "int-test-job"},
        status=200,
    )
    responses.add(
        responses.GET,
        f"{UNIPROT_API_URL}/idmapping/status/int-test-job",
        json={"jobStatus": "FINISHED"},
        status=200,
    )
    results = [
        {"from": eid, "to": {"primaryAccession": acc}}
        for eid, acc in mapping.items()
    ]
    responses.add(
        responses.GET,
        f"{UNIPROT_API_URL}/idmapping/uniprotkb/results/int-test-job",
        json={"results": results},
        status=200,
    )


def _prediction_json(uniprot_id: str, avg_plddt: float) -> list[dict]:
    """Build mock AlphaFold prediction API response."""
    return [
        {
            "uniprotAccession": uniprot_id,
            "confidenceAvgLocalScore": avg_plddt,
            "pdbUrl": f"{PDB_URL}/AF-{uniprot_id}-F1-model_v4.pdb",
            "latestVersion": 4,
        }
    ]


def _make_docking_df(n: int = 10) -> pd.DataFrame:
    """Build a docking DataFrame conforming to BATCH_DOCKING_COLUMNS."""
    return pd.DataFrame(
        {
            "drugbank_id": [f"DB{i:05d}" for i in range(n)],
            "drug_name": [f"Drug_{i}" for i in range(n)],
            "uniprot_id": ["P42858"] * n,
            "target_symbol": ["HTT"] * n,
            "affinity_kcal_mol": np.linspace(-9.5, -4.5, n),
            "output_file": [f"results/drug{i}_docked.pdbqt" for i in range(n)],
            "log_file": [f"results/drug{i}_log.txt" for i in range(n)],
        }
    )


def _make_confidence_list(uniprot_ids: list[str]) -> list[AlphaFoldConfidence]:
    """Build synthetic AlphaFold confidence records."""
    return [
        {
            "uniprot_id": uid,
            "avg_plddt": 85.0 + i,
            "model_url": f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb",
            "version": 4,
        }
        for i, uid in enumerate(uniprot_ids)
    ]


def _make_targets_df(
    uniprot_ids: list[str], symbols: list[str] | None = None
) -> pd.DataFrame:
    """Build a targets DataFrame conforming to TARGET_COLUMNS."""
    n = len(uniprot_ids)
    if symbols is None:
        symbols = [f"SYM{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "ensembl_id": [f"ENSG0000000{i:04d}" for i in range(n)],
            "symbol": symbols,
            "name": [f"Target protein {i}" for i in range(n)],
            "association_score": [0.9 - 0.05 * i for i in range(n)],
            "uniprot_id": uniprot_ids,
        }
    )


# ---------------------------------------------------------------------------
# Test: disease_targets -> alphafold_client data flow
# ---------------------------------------------------------------------------

# Reset the alphafold_client module-level session for ``responses`` mocking.
import drugsight.alphafold_client as _af_mod


@pytest.fixture(autouse=True)
def _reset_alphafold_session():
    """Clear the cached requests.Session so ``responses`` intercepts calls."""
    _af_mod._session = None
    yield
    _af_mod._session = None


class TestDiseaseToStructuresIntegration:
    """Verify disease_targets -> alphafold_client data flow."""

    @responses.activate
    def test_targets_feed_alphafold_batch(self, tmp_path: Path):
        """get_disease_targets output feeds directly into fetch_structures_batch."""
        from drugsight.alphafold_client import fetch_structures_batch
        from drugsight.disease_targets import get_disease_targets

        # -- Mock Open Targets --
        rows = [
            _target_row("ENSG00000197386", "HTT", "Huntingtin", 0.95),
            _target_row("ENSG00000176697", "BDNF", "BDNF protein", 0.82),
        ]
        responses.add(
            responses.POST,
            OPEN_TARGETS_URL,
            json=_open_targets_body(rows),
            status=200,
        )
        _register_uniprot_mocks({
            "ENSG00000197386": "P42858",
            "ENSG00000176697": "P23560",
        })

        # -- Mock AlphaFold confidence + PDB download --
        for uid, plddt in [("P42858", 88.0), ("P23560", 72.5)]:
            responses.add(
                responses.GET,
                f"{PREDICTION_URL}/{uid}",
                json=_prediction_json(uid, plddt),
                status=200,
            )
            filename = f"AF-{uid}-F1-model_v4.pdb"
            responses.add(
                responses.GET,
                f"{PDB_URL}/{filename}",
                body=SAMPLE_PDB_CONTENT,
                status=200,
                content_type="chemical/x-pdb",
            )

        # -- Execute the data flow --
        targets_df = get_disease_targets("MONDO_0007739", min_score=0.5)

        # Verify targets_df conforms to the schema contract.
        assert list(targets_df.columns) == TARGET_COLUMNS
        assert len(targets_df) == 2

        # Extract UniProt IDs the same way the pipeline does.
        uniprot_ids = targets_df["uniprot_id"].dropna().tolist()
        uniprot_ids = [uid for uid in uniprot_ids if uid]
        assert len(uniprot_ids) == 2

        # Feed into AlphaFold batch fetch.
        pdb_paths, confidences = fetch_structures_batch(
            uniprot_ids, output_dir=tmp_path, min_plddt=70.0
        )

        # Both proteins should pass the pLDDT filter.
        assert len(pdb_paths) == 2
        assert len(confidences) == 2
        for path in pdb_paths:
            assert path.exists()
            assert path.suffix == ".pdb"
        for conf in confidences:
            assert conf["avg_plddt"] >= 70.0

    @responses.activate
    def test_targets_with_low_plddt_filtered(self, tmp_path: Path):
        """When AlphaFold confidence is below threshold, structures are skipped."""
        from drugsight.alphafold_client import fetch_structures_batch
        from drugsight.disease_targets import get_disease_targets

        rows = [
            _target_row("ENSG00000000001", "LOW_CONF", "Low confidence target", 0.75),
        ]
        responses.add(
            responses.POST,
            OPEN_TARGETS_URL,
            json=_open_targets_body(rows),
            status=200,
        )
        _register_uniprot_mocks({"ENSG00000000001": "Q99999"})

        # Mock AlphaFold with a pLDDT below threshold.
        responses.add(
            responses.GET,
            f"{PREDICTION_URL}/Q99999",
            json=_prediction_json("Q99999", avg_plddt=45.0),
            status=200,
        )

        targets_df = get_disease_targets("MONDO_0007739", min_score=0.5)
        uniprot_ids = targets_df["uniprot_id"].dropna().tolist()

        pdb_paths, confidences = fetch_structures_batch(
            uniprot_ids, output_dir=tmp_path, min_plddt=70.0
        )

        # No structures should pass.
        assert len(pdb_paths) == 0
        assert len(confidences) == 0


# ---------------------------------------------------------------------------
# Test: docking -> ml_scorer data flow
# ---------------------------------------------------------------------------


class TestDockingToScoringIntegration:
    """Verify docking_engine output -> ml_scorer scoring data flow."""

    def test_docking_results_to_scored_candidates(self):
        """Feed sample docking results into merge_features -> score_candidates."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            merge_features,
            score_candidates,
        )

        docking_df = _make_docking_df(10)
        confidences = _make_confidence_list(["P42858"])
        targets_df = _make_targets_df(["P42858"], symbols=["HTT"])

        # Step 1: merge features from docking + confidence + targets.
        feature_df = merge_features(docking_df, confidences, targets_df)

        # Verify all SCORING_FEATURES are present and clean.
        for feat in SCORING_FEATURES:
            assert feat in feature_df.columns, f"Missing feature: {feat}"
            assert feature_df[feat].isna().sum() == 0, (
                f"Feature {feat} has NaN after merge"
            )

        # Step 2: score candidates.
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)

        assert "composite_score" in scored_df.columns
        assert "rank" in scored_df.columns
        assert scored_df["composite_score"].is_monotonic_decreasing

    def test_docking_to_explained_candidates(self):
        """Full flow: docking -> merge -> score -> explain."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            explain_top_candidates,
            merge_features,
            score_candidates,
        )

        docking_df = _make_docking_df(15)
        confidences = _make_confidence_list(["P42858"])
        targets_df = _make_targets_df(["P42858"], symbols=["HTT"])

        feature_df = merge_features(docking_df, confidences, targets_df)
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)
        ranked_df, shap_values = explain_top_candidates(scored_df, model, top_n=5)

        # Verify output has all expected columns.
        assert "composite_score" in ranked_df.columns
        assert "top_contributing_factor" in ranked_df.columns

        # Top 5 rows should have a valid contributing factor.
        top5_factors = ranked_df.head(5)["top_contributing_factor"]
        for factor in top5_factors:
            assert factor in SCORING_FEATURES, (
                f"Invalid top_contributing_factor: {factor}"
            )

        # SHAP shape must be (top_n, n_features).
        assert shap_values.shape == (5, len(SCORING_FEATURES))

    def test_docking_columns_survive_scoring(self):
        """Original BATCH_DOCKING_COLUMNS must be preserved through scoring."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            merge_features,
            score_candidates,
        )

        docking_df = _make_docking_df(5)
        feature_df = merge_features(docking_df, [], pd.DataFrame())
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)

        for col in BATCH_DOCKING_COLUMNS:
            assert col in scored_df.columns, (
                f"Docking column '{col}' lost after scoring pipeline"
            )

    def test_multi_target_docking_merge(self):
        """Merge handles docking results across multiple protein targets."""
        from drugsight.ml_scorer import _WeightedSumScorer, merge_features, score_candidates

        # Docking results against two different targets.
        df_a = _make_docking_df(5)
        df_a["uniprot_id"] = "P42858"
        df_a["target_symbol"] = "HTT"

        df_b = _make_docking_df(5)
        df_b["drugbank_id"] = [f"DB1{i:04d}" for i in range(5)]
        df_b["uniprot_id"] = "P23560"
        df_b["target_symbol"] = "BDNF"

        combined_docking = pd.concat([df_a, df_b], ignore_index=True)
        confidences = _make_confidence_list(["P42858", "P23560"])
        targets_df = _make_targets_df(
            ["P42858", "P23560"], symbols=["HTT", "BDNF"]
        )

        feature_df = merge_features(combined_docking, confidences, targets_df)

        assert len(feature_df) == 10
        # Both targets should have plddt_score populated from confidences.
        assert feature_df["plddt_score"].notna().all()
        # Both targets should have association_score from targets_df.
        assert feature_df["association_score"].notna().all()

        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)
        assert len(scored_df) == 10
        assert scored_df["composite_score"].notna().all()


# ---------------------------------------------------------------------------
# Test: ml_scorer with sample data files
# ---------------------------------------------------------------------------


class TestMLScorerWithSampleData:
    """Verify ml_scorer works with the actual sample data files in data/."""

    def test_train_on_sample_training_data(self, sample_training_df, tmp_path):
        """Train a model on the real training CSV and score sample docking data."""
        from drugsight.ml_scorer import merge_features, score_candidates, train_ranker

        # Write training data to tmp_path for train_ranker to consume.
        csv_path = tmp_path / "training.csv"
        sample_training_df.to_csv(csv_path, index=False)

        model = train_ranker(csv_path)
        assert hasattr(model, "predict")

        # Score using sample docking results.
        docking_df = _make_docking_df(8)
        confidences = _make_confidence_list(["P42858"])
        targets_df = _make_targets_df(["P42858"], symbols=["HTT"])

        feature_df = merge_features(docking_df, confidences, targets_df)
        scored_df = score_candidates(feature_df, model)

        assert "composite_score" in scored_df.columns
        assert scored_df["composite_score"].notna().all()
        assert len(scored_df) == 8

    def test_sample_docking_through_scorer(self, sample_docking_results_df):
        """Feed the real sample_docking_results.csv through the scorer."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            merge_features,
            score_candidates,
        )

        confidences = _make_confidence_list(
            sample_docking_results_df["uniprot_id"].unique().tolist()
        )
        targets_df = _make_targets_df(
            sample_docking_results_df["uniprot_id"].unique().tolist()
        )

        feature_df = merge_features(
            sample_docking_results_df, confidences, targets_df
        )
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)

        original_count = len(sample_docking_results_df)
        assert len(scored_df) == original_count
        assert scored_df["composite_score"].notna().all()
        assert scored_df["composite_score"].sum() > 0


# ---------------------------------------------------------------------------
# Test: full pipeline demo end-to-end
# ---------------------------------------------------------------------------


class TestFullPipelineDemoEndToEnd:
    """Full pipeline demo runs without error and produces valid output."""

    def test_end_to_end_valid_output(self):
        """Full pipeline demo produces a DataFrame with all expected columns."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

        # Verify all expected columns are present.
        for feat in SCORING_FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"
        assert "composite_score" in result.columns
        assert "top_contributing_factor" in result.columns

        # Verify scores are reasonable (not all NaN, not all identical).
        assert result["composite_score"].notna().all()
        assert result["composite_score"].nunique() > 1, (
            "composite_score values must not all be identical"
        )

    def test_end_to_end_rank_integrity(self):
        """Ranks are sequential integers starting at 1."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        assert "rank" in result.columns
        expected_ranks = list(range(1, len(result) + 1))
        assert list(result["rank"]) == expected_ranks

    def test_end_to_end_drug_identifiers(self):
        """Every row has recognizable DrugBank IDs and drug names."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        # DrugBank IDs should start with "DB".
        for db_id in result["drugbank_id"]:
            assert db_id.startswith("DB"), f"Unexpected drugbank_id format: {db_id}"

        # Drug names should be non-empty strings.
        assert (result["drug_name"].str.len() > 0).all()

    def test_end_to_end_target_info_preserved(self):
        """Docking metadata (uniprot_id, target_symbol) is carried through."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        assert "uniprot_id" in result.columns
        assert "target_symbol" in result.columns
        assert result["uniprot_id"].notna().all()
        assert result["target_symbol"].notna().all()

    def test_end_to_end_binding_affinity_derived(self):
        """binding_affinity feature is derived (abs of affinity_kcal_mol)."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        # binding_affinity should be >= 0 (it is abs(affinity_kcal_mol)).
        assert (result["binding_affinity"] >= 0).all(), (
            "binding_affinity must be non-negative (absolute value of affinity)"
        )

    def test_end_to_end_feature_ranges(self):
        """Scoring features fall within expected ranges after pipeline demo."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        # plddt_score should be in [0, 100] range.
        assert (result["plddt_score"] >= 0).all()
        assert (result["plddt_score"] <= 100).all()

        # association_score should be in [0, 1] range.
        assert (result["association_score"] >= 0).all()
        assert (result["association_score"] <= 1).all()

        # safety_score should be in [0, 1] range.
        assert (result["safety_score"] >= 0).all()
        assert (result["safety_score"] <= 1).all()

        # patent_expired should be 0 or 1.
        assert set(result["patent_expired"].unique()).issubset({0, 0.0, 1, 1.0})
