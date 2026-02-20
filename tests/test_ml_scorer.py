"""Unit tests for drugsight.ml_scorer — ML-Based Multi-Factor Scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drugsight.schemas import (
    BATCH_DOCKING_COLUMNS,
    SCORING_FEATURES,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_docking_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal docking DataFrame conforming to BATCH_DOCKING_COLUMNS."""
    return pd.DataFrame(
        {
            "drugbank_id": [f"DB{i:05d}" for i in range(n)],
            "drug_name": [f"Drug_{i}" for i in range(n)],
            "uniprot_id": ["P42858"] * n,
            "target_symbol": ["HTT"] * n,
            "affinity_kcal_mol": np.linspace(-9.0, -5.0, n),
            "output_file": [f"results/drug{i}_docked.pdbqt" for i in range(n)],
            "log_file": [f"results/drug{i}_log.txt" for i in range(n)],
        }
    )


def _make_confidence_list(uniprot_ids: list[str]) -> list[dict]:
    """Build a sample AlphaFoldConfidence list."""
    return [
        {"uniprot_id": uid, "avg_plddt": 85.0 + i, "model_url": "", "version": 4}
        for i, uid in enumerate(uniprot_ids)
    ]


def _make_association_df(uniprot_ids: list[str]) -> pd.DataFrame:
    """Build a minimal association DataFrame."""
    return pd.DataFrame(
        {
            "uniprot_id": uniprot_ids,
            "association_score": [0.7 + 0.01 * i for i in range(len(uniprot_ids))],
        }
    )


def _make_feature_df(n: int = 5) -> pd.DataFrame:
    """Build a feature DataFrame with all SCORING_FEATURES populated."""
    rng = np.random.default_rng(42)
    data: dict[str, list] = {}
    for col in BATCH_DOCKING_COLUMNS:
        if col == "affinity_kcal_mol":
            data[col] = list(np.linspace(-9.0, -5.0, n))
        elif col in ("drugbank_id", "drug_name", "target_symbol", "output_file", "log_file"):
            data[col] = [f"{col}_{i}" for i in range(n)]
        elif col == "uniprot_id":
            data[col] = ["P42858"] * n
        else:
            data[col] = [0.0] * n

    for feat in SCORING_FEATURES:
        if feat == "patent_expired":
            data[feat] = pd.array(rng.integers(0, 2, size=n).tolist(), dtype="Int64")
        elif feat == "bioactivity_count":
            data[feat] = list(rng.integers(0, 20, size=n))
        else:
            data[feat] = list(rng.uniform(0.1, 1.0, size=n))

    return pd.DataFrame(data)


def _write_training_csv(path, n_rows: int) -> None:
    """Write a minimal training CSV with the required schema."""
    rng = np.random.default_rng(99)
    rows: dict[str, list] = {
        "disease_id": [f"EFO_{i:07d}" for i in range(n_rows)],
        "drugbank_id": [f"DB{i:05d}" for i in range(n_rows)],
        "drug_name": [f"Drug_{i}" for i in range(n_rows)],
        "repurposing_success": list(rng.integers(0, 2, size=n_rows)),
    }
    for feat in SCORING_FEATURES:
        if feat == "patent_expired":
            rows[feat] = list(rng.integers(0, 2, size=n_rows))
        elif feat == "bioactivity_count":
            rows[feat] = list(rng.integers(0, 20, size=n_rows))
        else:
            rows[feat] = list(rng.uniform(0.1, 1.0, size=n_rows))

    pd.DataFrame(rows).to_csv(path, index=False)


# ── Tests ────────────────────────────────────────────────────────────────


class TestTrainRanker:
    """Tests for train_ranker()."""

    def test_train_ranker_returns_model(self, sample_training_df: pd.DataFrame, tmp_path):
        """Train on training_repurposing_cases.csv, verify model has .predict() method."""
        from drugsight.ml_scorer import train_ranker

        # sample_training_df is loaded from the real CSV; write it out so
        # train_ranker can read it as a file.
        csv_path = tmp_path / "training.csv"
        sample_training_df.to_csv(csv_path, index=False)

        model = train_ranker(csv_path)

        assert hasattr(model, "predict"), "Model must expose a .predict() method"

        # Verify predict produces correct output shape.
        X = sample_training_df[SCORING_FEATURES].astype(np.float64).head(5)
        predictions = model.predict(X)
        assert len(predictions) == 5, "predict() output length must match input"

    def test_train_ranker_small_data_fallback(self, tmp_path):
        """Create tiny CSV (10 rows), verify falls back to weighted sum scorer (no crash)."""
        from drugsight.ml_scorer import _WeightedSumScorer, train_ranker

        csv_path = tmp_path / "tiny_training.csv"
        _write_training_csv(csv_path, n_rows=10)

        model = train_ranker(csv_path)

        assert isinstance(model, _WeightedSumScorer), (
            "With < 50 rows, train_ranker must fall back to _WeightedSumScorer"
        )
        assert hasattr(model, "predict")

    def test_train_ranker_missing_file_fallback(self, tmp_path):
        """Non-existent CSV path should return fallback scorer, not raise."""
        from drugsight.ml_scorer import _WeightedSumScorer, train_ranker

        model = train_ranker(tmp_path / "nonexistent.csv")
        assert isinstance(model, _WeightedSumScorer)

    def test_train_ranker_missing_columns_raises(self, tmp_path):
        """CSV missing required columns should raise ValueError."""
        from drugsight.ml_scorer import train_ranker

        # Write a CSV with enough rows but missing SCORING_FEATURES.
        bad_df = pd.DataFrame(
            {
                "disease_id": [f"EFO_{i}" for i in range(60)],
                "repurposing_success": [1] * 60,
                "binding_affinity": [7.0] * 60,
                # Deliberately omit the other SCORING_FEATURES.
            }
        )
        csv_path = tmp_path / "bad_training.csv"
        bad_df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            train_ranker(csv_path)


class TestMergeFeatures:
    """Tests for merge_features()."""

    def test_merge_features_column_mapping(self):
        """Pass sample docking_df + confidence + association data.
        Verify binding_affinity mapped from affinity_kcal_mol.
        Verify plddt_score from avg_plddt.
        Verify all SCORING_FEATURES present."""
        from drugsight.ml_scorer import merge_features

        docking_df = _make_docking_df(5)
        confidence = _make_confidence_list(["P42858"])
        association = _make_association_df(["P42858"])

        result = merge_features(docking_df, confidence, association)

        assert "binding_affinity" in result.columns, (
            "binding_affinity must be mapped from affinity_kcal_mol"
        )
        assert "plddt_score" in result.columns, (
            "plddt_score must be mapped from avg_plddt"
        )

        for feat in SCORING_FEATURES:
            assert feat in result.columns, f"Missing scoring feature: {feat}"

    def test_merge_features_binding_affinity_values(self):
        """Verify binding_affinity is the absolute value of affinity_kcal_mol."""
        from drugsight.ml_scorer import merge_features

        docking_df = _make_docking_df(3)
        result = merge_features(docking_df, [], pd.DataFrame())

        expected = docking_df["affinity_kcal_mol"].abs()
        pd.testing.assert_series_equal(
            result["binding_affinity"],
            expected,
            check_names=False,
        )

    def test_merge_features_fills_missing(self):
        """Pass incomplete data, verify missing features filled with defaults (not NaN)."""
        from drugsight.ml_scorer import merge_features

        docking_df = _make_docking_df(3)
        # Pass empty confidence and association — many features will be missing.
        result = merge_features(docking_df, [], pd.DataFrame())

        for feat in SCORING_FEATURES:
            assert feat in result.columns, f"Missing feature column: {feat}"
            assert result[feat].isna().sum() == 0, (
                f"Feature {feat} has NaN values — should be filled with defaults"
            )

    def test_merge_features_preserves_docking_columns(self):
        """Original BATCH_DOCKING_COLUMNS must survive the merge."""
        from drugsight.ml_scorer import merge_features

        docking_df = _make_docking_df(3)
        result = merge_features(docking_df, [], pd.DataFrame())

        for col in BATCH_DOCKING_COLUMNS:
            assert col in result.columns, f"Lost docking column after merge: {col}"

    def test_merge_features_plddt_join(self):
        """Verify plddt_score is populated from confidence_list avg_plddt."""
        from drugsight.ml_scorer import merge_features

        docking_df = _make_docking_df(3)
        confidence = [
            {"uniprot_id": "P42858", "avg_plddt": 92.5, "model_url": "", "version": 4},
        ]
        result = merge_features(docking_df, confidence, pd.DataFrame())

        assert (result["plddt_score"] == 92.5).all(), (
            "plddt_score must be populated from the confidence list avg_plddt"
        )

    def test_merge_features_association_join(self):
        """Verify association_score is populated from association_df."""
        from drugsight.ml_scorer import merge_features

        docking_df = _make_docking_df(3)
        assoc_df = pd.DataFrame(
            {"uniprot_id": ["P42858"], "association_score": [0.88]}
        )
        result = merge_features(docking_df, [], assoc_df)

        assert (result["association_score"] == 0.88).all()


class TestScoreCandidates:
    """Tests for score_candidates()."""

    def test_score_candidates_adds_columns(self):
        """Pass feature_df + model, verify composite_score and rank columns added."""
        from drugsight.ml_scorer import _WeightedSumScorer, score_candidates

        feature_df = _make_feature_df(10)
        model = _WeightedSumScorer()

        result = score_candidates(feature_df, model)

        assert "composite_score" in result.columns, "composite_score column missing"
        assert "rank" in result.columns, "rank column missing"
        assert len(result) == 10

    def test_score_candidates_rank_ordering(self):
        """Verify rank 1 has highest composite_score."""
        from drugsight.ml_scorer import _WeightedSumScorer, score_candidates

        feature_df = _make_feature_df(20)
        model = _WeightedSumScorer()

        result = score_candidates(feature_df, model)

        # Rank 1 should be the first row (highest score).
        assert result.iloc[0]["rank"] == 1
        assert result["composite_score"].is_monotonic_decreasing, (
            "Rows must be sorted by composite_score descending"
        )

        # Ranks should be sequential 1..N.
        expected_ranks = list(range(1, len(result) + 1))
        assert list(result["rank"]) == expected_ranks

    def test_score_candidates_preserves_features(self):
        """Original SCORING_FEATURES columns must survive scoring."""
        from drugsight.ml_scorer import _WeightedSumScorer, score_candidates

        feature_df = _make_feature_df(5)
        model = _WeightedSumScorer()

        result = score_candidates(feature_df, model)

        for feat in SCORING_FEATURES:
            assert feat in result.columns, f"Feature {feat} lost after scoring"


class TestExplainTopCandidates:
    """Tests for explain_top_candidates()."""

    def test_explain_top_candidates_shap(self):
        """Verify top_contributing_factor is one of SCORING_FEATURES."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            explain_top_candidates,
            score_candidates,
        )

        feature_df = _make_feature_df(15)
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)

        ranked_df, shap_values = explain_top_candidates(scored_df, model, top_n=5)

        # Check that top_contributing_factor values are valid feature names.
        top_rows = ranked_df.head(5)
        for factor in top_rows["top_contributing_factor"]:
            assert factor in SCORING_FEATURES, (
                f"top_contributing_factor '{factor}' is not in SCORING_FEATURES"
            )

    def test_explain_returns_tuple(self):
        """Verify returns (DataFrame, array-like) tuple."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            explain_top_candidates,
            score_candidates,
        )

        feature_df = _make_feature_df(10)
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)

        result = explain_top_candidates(scored_df, model, top_n=3)

        assert isinstance(result, tuple), "explain_top_candidates must return a tuple"
        assert len(result) == 2, "Tuple must contain exactly 2 elements"

        ranked_df, shap_values = result
        assert isinstance(ranked_df, pd.DataFrame), "First element must be a DataFrame"
        assert hasattr(shap_values, "__len__"), "Second element must be array-like"

    def test_explain_shap_values_shape(self):
        """SHAP values array must have shape (top_n, n_features)."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            explain_top_candidates,
            score_candidates,
        )

        top_n = 5
        feature_df = _make_feature_df(20)
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)

        _, shap_values = explain_top_candidates(scored_df, model, top_n=top_n)

        assert shap_values.shape == (top_n, len(SCORING_FEATURES)), (
            f"Expected SHAP shape ({top_n}, {len(SCORING_FEATURES)}), "
            f"got {shap_values.shape}"
        )

    def test_explain_without_prior_scoring(self):
        """explain_top_candidates should work even if rank column is missing
        (it calls score_candidates internally)."""
        from drugsight.ml_scorer import _WeightedSumScorer, explain_top_candidates

        feature_df = _make_feature_df(10)
        model = _WeightedSumScorer()

        # No prior score_candidates call — rank column is absent.
        assert "rank" not in feature_df.columns

        ranked_df, shap_values = explain_top_candidates(feature_df, model, top_n=3)

        assert "rank" in ranked_df.columns, "explain should add rank if missing"
        assert "top_contributing_factor" in ranked_df.columns

    def test_explain_non_top_rows_empty_factor(self):
        """Rows beyond top_n should have empty top_contributing_factor."""
        from drugsight.ml_scorer import (
            _WeightedSumScorer,
            explain_top_candidates,
            score_candidates,
        )

        feature_df = _make_feature_df(10)
        model = _WeightedSumScorer()
        scored_df = score_candidates(feature_df, model)

        ranked_df, _ = explain_top_candidates(scored_df, model, top_n=3)

        # Rows beyond the top 3 should have empty string for the factor.
        tail_factors = ranked_df.iloc[3:]["top_contributing_factor"]
        assert (tail_factors == "").all(), (
            "Rows beyond top_n must have empty top_contributing_factor"
        )


class TestNullableIntHandling:
    """Tests for nullable integer (Int64) handling with NaN values."""

    def test_nullable_int_handling(self):
        """Create DataFrame with NaN in patent_expired, verify no Int casting errors.
        Use 'Int64' (capital I) for nullable ints."""
        from drugsight.ml_scorer import _WeightedSumScorer, score_candidates

        feature_df = _make_feature_df(5)

        # Introduce NaN into patent_expired using nullable Int64.
        feature_df["patent_expired"] = pd.array([1, 0, pd.NA, 1, pd.NA], dtype="Int64")

        # Fill NaN before scoring (as merge_features would do).
        feature_df["patent_expired"] = feature_df["patent_expired"].fillna(0)

        model = _WeightedSumScorer()

        # This should not raise IntCastingNaNError.
        result = score_candidates(feature_df, model)
        assert "composite_score" in result.columns
        assert len(result) == 5

    def test_merge_features_handles_nullable_int(self):
        """merge_features must fill NaN in patent_expired without IntCastingNaNError."""
        from drugsight.ml_scorer import merge_features

        docking_df = _make_docking_df(4)
        # Simulate confidence data that might leave NaN gaps after a left join.
        confidence = _make_confidence_list(["XXXXXX"])  # No match with P42858
        association = _make_association_df(["XXXXXX"])   # No match with P42858

        # This should not crash even though most features will be filled from defaults.
        result = merge_features(docking_df, confidence, association)

        assert result["patent_expired"].isna().sum() == 0, (
            "patent_expired must not contain NaN after merge_features"
        )


class TestWeightedSumScorer:
    """Tests for the _WeightedSumScorer fallback."""

    def test_weighted_sum_scorer_predict(self):
        """Directly test the fallback scorer's .predict() returns array of correct length."""
        from drugsight.ml_scorer import _WeightedSumScorer

        model = _WeightedSumScorer()
        feature_df = _make_feature_df(8)

        predictions = model.predict(feature_df[SCORING_FEATURES])

        assert isinstance(predictions, np.ndarray), "predict() must return numpy array"
        assert predictions.shape == (8,), f"Expected shape (8,), got {predictions.shape}"
        assert predictions.dtype == np.float64

    def test_weighted_sum_scorer_predict_numpy_input(self):
        """Verify .predict() also works with raw numpy array input."""
        from drugsight.ml_scorer import _WeightedSumScorer

        model = _WeightedSumScorer()
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, size=(6, len(SCORING_FEATURES)))

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (6,)

    def test_weighted_sum_scorer_weights_sum_to_one(self):
        """Model weights should sum to 1.0 for interpretable composite scores."""
        from drugsight.ml_scorer import _WeightedSumScorer

        model = _WeightedSumScorer()
        total_weight = sum(model._weights.values())
        assert abs(total_weight - 1.0) < 1e-9, (
            f"Weights sum to {total_weight}, expected 1.0"
        )

    def test_weighted_sum_scorer_all_features_have_weights(self):
        """Every SCORING_FEATURE should have a corresponding weight."""
        from drugsight.ml_scorer import _WeightedSumScorer

        model = _WeightedSumScorer()
        for feat in SCORING_FEATURES:
            assert feat in model._weights, (
                f"Feature {feat} has no weight in _WeightedSumScorer"
            )

    def test_weighted_sum_scorer_deterministic(self):
        """Same input should always produce the same output."""
        from drugsight.ml_scorer import _WeightedSumScorer

        model = _WeightedSumScorer()
        feature_df = _make_feature_df(5)
        X = feature_df[SCORING_FEATURES]

        result1 = model.predict(X)
        result2 = model.predict(X)

        np.testing.assert_array_equal(result1, result2)
