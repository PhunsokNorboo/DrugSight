"""
Pipeline orchestrator tests — drugsight.pipeline.

Tests the demo pipeline end-to-end using pre-computed sample data from
``data/``.  These tests exercise the scoring/ranking stages without any
external API access, AutoDock Vina, or RDKit installation.

The full ``run_pipeline()`` (which chains all six modules including live
API calls and docking) is not tested here — see ``test_integration.py``
for mock-based integration tests that validate module-to-module data flow.
"""

from __future__ import annotations

import pandas as pd
import pytest

from drugsight.schemas import RANKED_RESULT_COLUMNS, SCORING_FEATURES


# ---------------------------------------------------------------------------
# run_pipeline_demo() — returns ranked DataFrame
# ---------------------------------------------------------------------------


class TestRunPipelineDemoResults:
    """Verify run_pipeline_demo() produces a well-formed ranked DataFrame."""

    def test_returns_dataframe(self):
        """run_pipeline_demo() returns a DataFrame with ranked results."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "composite_score" in result.columns
        assert "rank" in result.columns
        assert "drug_name" in result.columns

    def test_has_scoring_features(self):
        """Demo results include all scoring features defined in schemas."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        for feat in SCORING_FEATURES:
            assert feat in result.columns, f"Missing scoring feature: {feat}"

    def test_has_ranked_result_columns(self):
        """Demo results include every column declared in RANKED_RESULT_COLUMNS."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        for col in RANKED_RESULT_COLUMNS:
            assert col in result.columns, f"Missing RANKED_RESULT_COLUMNS entry: {col}"

    def test_sorted_by_score_descending(self):
        """Results are sorted by composite_score descending."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()
        scores = result["composite_score"].tolist()

        assert scores == sorted(scores, reverse=True), (
            "Ranked candidates must be sorted by composite_score descending"
        )

    def test_scores_are_valid(self):
        """Composite scores are all finite, non-NaN, and not all identical."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        assert result["composite_score"].notna().all(), (
            "composite_score must not contain NaN values"
        )
        # Scores from a ranking model can be negative; verify they are not
        # all the same value (which would indicate a degenerate model).
        assert result["composite_score"].nunique() > 1, (
            "composite_score values must not all be identical"
        )

    def test_top_contributing_factor_is_valid_feature(self):
        """top_contributing_factor for ranked rows must be a known SCORING_FEATURE."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        # Only non-empty entries should be validated (rows beyond top_n
        # get an empty string from explain_top_candidates).
        explained = result.loc[result["top_contributing_factor"] != ""]
        assert len(explained) > 0, "At least some rows should have explanations"

        for factor in explained["top_contributing_factor"]:
            assert factor in SCORING_FEATURES, (
                f"top_contributing_factor '{factor}' not in SCORING_FEATURES"
            )

    def test_returns_at_most_20_rows(self):
        """Demo caps output at 20 candidates."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()
        assert len(result) <= 20

    def test_drug_identifiers_non_empty(self):
        """Every row must have a non-empty drugbank_id and drug_name."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        assert result["drugbank_id"].notna().all()
        assert (result["drugbank_id"].str.len() > 0).all()
        assert result["drug_name"].notna().all()
        assert (result["drug_name"].str.len() > 0).all()


# ---------------------------------------------------------------------------
# run_pipeline_demo() — intermediate file saves
# ---------------------------------------------------------------------------


class TestRunPipelineDemoSavesIntermediates:
    """Verify the demo pipeline writes results to disk."""

    def test_saves_ranked_candidates_csv(self, tmp_path, monkeypatch):
        """Pipeline saves ranked_candidates.csv to the results directory."""
        import drugsight.pipeline as pipeline_mod

        # Redirect RESULTS_DIR to tmp_path so we don't pollute the real dir.
        monkeypatch.setattr(pipeline_mod, "RESULTS_DIR", tmp_path)

        pipeline_mod.run_pipeline_demo()

        output_csv = tmp_path / "ranked_candidates.csv"
        assert output_csv.exists(), "ranked_candidates.csv was not created"

        # Verify the CSV is readable and non-empty.
        saved_df = pd.read_csv(output_csv)
        assert len(saved_df) > 0
        assert "composite_score" in saved_df.columns

    def test_creates_results_dir_if_missing(self, tmp_path, monkeypatch):
        """Pipeline creates the results directory when it does not exist."""
        import drugsight.pipeline as pipeline_mod

        new_dir = tmp_path / "fresh_results"
        assert not new_dir.exists()

        monkeypatch.setattr(pipeline_mod, "RESULTS_DIR", new_dir)
        pipeline_mod.run_pipeline_demo()

        assert new_dir.exists()
        assert (new_dir / "ranked_candidates.csv").exists()


# ---------------------------------------------------------------------------
# run_pipeline_demo() — reproducibility
# ---------------------------------------------------------------------------


class TestRunPipelineDemoReproducibility:
    """Verify the demo is deterministic across calls."""

    def test_deterministic_output(self):
        """Two consecutive demo runs produce identical rankings."""
        from drugsight.pipeline import run_pipeline_demo

        result_a = run_pipeline_demo()
        result_b = run_pipeline_demo()

        pd.testing.assert_frame_equal(result_a, result_b)

    def test_custom_disease_id_label(self):
        """Passing a different disease_id does not crash (label-only param)."""
        from drugsight.pipeline import run_pipeline_demo

        # The demo ignores the disease_id for data loading; it is only a label.
        result = run_pipeline_demo(disease_id="EFO_0000000")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# run_pipeline_demo() — scoring features have no NaN
# ---------------------------------------------------------------------------


class TestRunPipelineDemoFeatureQuality:
    """Verify feature columns are clean after the demo pipeline runs."""

    def test_no_nan_in_scoring_features(self):
        """All SCORING_FEATURES columns must be free of NaN."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()

        for feat in SCORING_FEATURES:
            nan_count = result[feat].isna().sum()
            assert nan_count == 0, (
                f"Feature '{feat}' has {nan_count} NaN value(s) in demo output"
            )

    def test_rank_column_sequential(self):
        """Rank column must be a sequential 1..N integer series."""
        from drugsight.pipeline import run_pipeline_demo

        result = run_pipeline_demo()
        expected_ranks = list(range(1, len(result) + 1))
        assert list(result["rank"]) == expected_ranks
