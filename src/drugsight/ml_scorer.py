"""
ML-Based Multi-Factor Scoring — Module 5 of DrugSight.

Trains a LightGBM LambdaMART learning-to-rank model on known drug
repurposing cases, then uses it to score and rank new candidates.
Provides SHAP-based explanations for why each drug ranks high.

When training data is insufficient (< 50 rows), falls back to a
hand-tuned weighted-sum scorer with the same ``.predict(X)`` interface
so downstream code never needs to branch on model type.

All heavy dependencies (lightgbm, shap, numpy) are lazy-imported to
keep ``--dry-run`` and unit tests fast when these packages are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from drugsight.config import DATA_DIR
from drugsight.schemas import (
    BATCH_DOCKING_COLUMNS,
    SCORING_FEATURES,
)

logger = logging.getLogger(__name__)

# ── Feature-level defaults (used when values are missing) ────────────────
_FEATURE_DEFAULTS: dict[str, float] = {
    "binding_affinity": 0.5,
    "plddt_score": 0.5,
    "bioactivity_count": 0,
    "safety_score": 0.5,
    "tanimoto_similarity": 0.5,
    "association_score": 0.5,
    "patent_expired": 0,
    "oral_bioavailability": 0.5,
}

# ── Minimum rows required for LambdaMART training ───────────────────────
_MIN_TRAINING_ROWS = 50


# ── Weighted-sum fallback ────────────────────────────────────────────────

class _WeightedSumScorer:
    """Fallback scorer used when training data is too small for LambdaMART.

    Implements the same ``.predict(X)`` interface as ``lgb.LGBMRanker``
    so callers never need to branch on model type.
    """

    _weights: dict[str, float] = {
        "binding_affinity": 0.25,
        "plddt_score": 0.10,
        "bioactivity_count": 0.15,
        "safety_score": 0.10,
        "tanimoto_similarity": 0.15,
        "association_score": 0.15,
        "patent_expired": 0.05,
        "oral_bioavailability": 0.05,
    }

    def predict(self, X: pd.DataFrame) -> Any:
        """Return weighted-sum scores matching LGBMRanker's output shape.

        Parameters
        ----------
        X:
            DataFrame (or array-like) with columns matching
            ``SCORING_FEATURES``.

        Returns
        -------
        numpy.ndarray of shape ``(n_samples,)`` with composite scores.
        """
        import numpy as np

        if isinstance(X, pd.DataFrame):
            scores = np.zeros(len(X), dtype=np.float64)
            for feature, weight in self._weights.items():
                if feature in X.columns:
                    scores += weight * X[feature].to_numpy(dtype=np.float64)
            return scores

        # Fall back for numpy array input (column order = SCORING_FEATURES).
        X_arr = np.asarray(X, dtype=np.float64)
        weight_vec = np.array(
            [self._weights.get(f, 0.0) for f in SCORING_FEATURES],
            dtype=np.float64,
        )
        return X_arr @ weight_vec


# ── Public API ───────────────────────────────────────────────────────────


def train_ranker(training_csv: str | Path) -> Any:
    """Train an LGBMRanker on known repurposing cases.

    Falls back to :class:`_WeightedSumScorer` when the training set has
    fewer than ``_MIN_TRAINING_ROWS`` rows.

    Parameters
    ----------
    training_csv:
        Path to a CSV file with columns: all ``SCORING_FEATURES`` +
        ``disease_id`` + ``repurposing_success`` (0/1 label).

    Returns
    -------
    A model object with a ``.predict(X)`` method.  Either an
    ``lgb.LGBMRanker`` or a ``_WeightedSumScorer``.
    """
    import numpy as np

    csv_path = Path(training_csv)
    if not csv_path.exists():
        logger.warning(
            "Training CSV not found at %s — returning weighted-sum fallback",
            csv_path,
        )
        return _WeightedSumScorer()

    df = pd.read_csv(csv_path)
    logger.info("Loaded training data: %d rows from %s", len(df), csv_path)

    # Validate required columns.
    required = set(SCORING_FEATURES) | {"disease_id", "repurposing_success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Training CSV is missing required columns: {sorted(missing)}"
        )

    # Fall back to weighted sum if data is too small for LambdaMART.
    if len(df) < _MIN_TRAINING_ROWS:
        logger.warning(
            "Only %d training rows (need >= %d) — using weighted-sum fallback",
            len(df),
            _MIN_TRAINING_ROWS,
        )
        return _WeightedSumScorer()

    # ── Prepare LambdaMART inputs ────────────────────────────────────────
    import lightgbm as lgb

    # Sort by disease_id so group boundaries are contiguous.
    df = df.sort_values("disease_id").reset_index(drop=True)

    X = df[SCORING_FEATURES].astype(np.float64)
    y = df["repurposing_success"].astype(np.float64)

    # LambdaMART requires a groups array: number of items per query group.
    group_sizes = df.groupby("disease_id", sort=False).size().to_numpy()

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
    )

    logger.info(
        "Training LGBMRanker: %d rows, %d disease groups",
        len(df),
        len(group_sizes),
    )
    ranker.fit(X, y, group=group_sizes)

    logger.info("LGBMRanker training complete")
    return ranker


def merge_features(
    docking_df: pd.DataFrame,
    confidence_list: list[dict],
    association_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join docking results, AlphaFold confidence, and target associations.

    Produces a unified feature matrix ready for scoring.

    Parameters
    ----------
    docking_df:
        DataFrame with columns matching ``BATCH_DOCKING_COLUMNS``.
        Must include ``affinity_kcal_mol`` and ``uniprot_id``.
    confidence_list:
        List of ``AlphaFoldConfidence`` dicts, each containing at least
        ``uniprot_id`` and ``avg_plddt``.
    association_df:
        DataFrame with at least ``uniprot_id`` and ``association_score``
        columns (or ``ensembl_id`` if ``uniprot_id`` is unavailable).

    Returns
    -------
    pd.DataFrame
        Feature matrix with all ``BATCH_DOCKING_COLUMNS`` plus all
        ``SCORING_FEATURES`` populated.
    """
    # Start from docking results.
    df = docking_df.copy()

    # Map affinity_kcal_mol → binding_affinity (negate so higher = better).
    df["binding_affinity"] = df["affinity_kcal_mol"].abs()

    # ── Join AlphaFold pLDDT confidence ──────────────────────────────────
    if confidence_list:
        conf_df = pd.DataFrame(confidence_list)
        if "avg_plddt" in conf_df.columns and "uniprot_id" in conf_df.columns:
            conf_lookup = conf_df[["uniprot_id", "avg_plddt"]].drop_duplicates(
                subset="uniprot_id", keep="first"
            )
            df = df.merge(
                conf_lookup,
                on="uniprot_id",
                how="left",
            )
            df["plddt_score"] = df["avg_plddt"]
            df = df.drop(columns=["avg_plddt"], errors="ignore")
        else:
            logger.warning(
                "Confidence list missing expected keys — skipping pLDDT join"
            )

    # ── Join association scores ──────────────────────────────────────────
    if not association_df.empty:
        # Determine join key: prefer uniprot_id, fall back to ensembl_id.
        if "uniprot_id" in association_df.columns:
            join_key = "uniprot_id"
        elif "ensembl_id" in association_df.columns and "ensembl_id" in df.columns:
            join_key = "ensembl_id"
        else:
            join_key = None
            logger.warning(
                "No common join key between docking_df and association_df — "
                "skipping association_score join"
            )

        if join_key is not None:
            assoc_cols = [join_key, "association_score"]
            assoc_subset = (
                association_df[assoc_cols]
                .drop_duplicates(subset=join_key, keep="first")
            )

            # Avoid column collision if association_score already exists
            # (e.g., from a prior merge step).
            if "association_score" in df.columns:
                df = df.drop(columns=["association_score"])

            df = df.merge(assoc_subset, on=join_key, how="left")

    # ── Fill missing scoring features with sensible defaults ─────────────
    for feature in SCORING_FEATURES:
        if feature not in df.columns:
            df[feature] = _FEATURE_DEFAULTS[feature]
        else:
            df[feature] = df[feature].fillna(_FEATURE_DEFAULTS[feature])

    logger.info(
        "Feature matrix built: %d candidates, %d features",
        len(df),
        len(SCORING_FEATURES),
    )
    return df


def score_candidates(
    feature_df: pd.DataFrame,
    model: Any,
) -> pd.DataFrame:
    """Score and rank candidates using a trained model.

    Parameters
    ----------
    feature_df:
        DataFrame containing all ``SCORING_FEATURES`` columns.
    model:
        A model with a ``.predict(X)`` method — either an
        ``LGBMRanker`` or a ``_WeightedSumScorer``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with ``composite_score`` and ``rank``
        columns, sorted by ``composite_score`` descending (rank 1 = best).
    """
    import numpy as np

    X = feature_df[SCORING_FEATURES].astype(np.float64)
    scores = model.predict(X)

    df = feature_df.copy()
    df["composite_score"] = scores
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    logger.info(
        "Scored %d candidates — top composite_score: %.4f",
        len(df),
        df["composite_score"].iloc[0] if len(df) > 0 else 0.0,
    )
    return df


def explain_top_candidates(
    feature_df: pd.DataFrame,
    model: Any,
    top_n: int = 10,
) -> tuple[pd.DataFrame, Any]:
    """Add SHAP-based explanations to the top-ranked candidates.

    Parameters
    ----------
    feature_df:
        Scored DataFrame (must already have ``composite_score`` and
        ``rank`` columns from :func:`score_candidates`).
    model:
        The model used for scoring.
    top_n:
        Number of top candidates to explain.

    Returns
    -------
    tuple of (ranked_df, shap_values)
        ``ranked_df`` is the input DataFrame with an added
        ``top_contributing_factor`` column.
        ``shap_values`` is a numpy array of shape ``(top_n, n_features)``
        containing the SHAP (or pseudo-SHAP) values for the top
        candidates.
    """
    import numpy as np

    df = feature_df.copy()

    # Ensure we have a rank column; if not, score first.
    if "rank" not in df.columns:
        df = score_candidates(df, model)

    df = df.sort_values("rank").reset_index(drop=True)
    top = df.head(top_n)
    X_top = top[SCORING_FEATURES].astype(np.float64)

    # ── Compute SHAP values ──────────────────────────────────────────────
    if isinstance(model, _WeightedSumScorer):
        # Pseudo-SHAP: weight * feature_value per feature.
        shap_values = np.zeros((len(X_top), len(SCORING_FEATURES)), dtype=np.float64)
        for col_idx, feature in enumerate(SCORING_FEATURES):
            weight = model._weights.get(feature, 0.0)
            shap_values[:, col_idx] = weight * X_top[feature].to_numpy(dtype=np.float64)

        logger.info(
            "Computed pseudo-SHAP values for top %d candidates (fallback model)",
            len(X_top),
        )
    else:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_result = explainer.shap_values(X_top)

        # shap_values may be returned as a list (one per class) or a
        # single array.  For ranking models it is typically a single array.
        if isinstance(shap_result, list):
            shap_values = np.asarray(shap_result[0], dtype=np.float64)
        else:
            shap_values = np.asarray(shap_result, dtype=np.float64)

        logger.info(
            "Computed SHAP values for top %d candidates (TreeExplainer)",
            len(X_top),
        )

    # ── Identify top contributing factor per candidate ────────────────────
    top_factor_indices = np.abs(shap_values).argmax(axis=1)
    top_factors = [SCORING_FEATURES[i] for i in top_factor_indices]

    # Write top_contributing_factor back into the full DataFrame.
    # Only the top_n rows get an explanation; others get an empty string.
    df["top_contributing_factor"] = ""
    df.loc[df.index[:top_n], "top_contributing_factor"] = top_factors

    logger.info(
        "Top contributing factors for top %d: %s",
        top_n,
        ", ".join(dict.fromkeys(top_factors)),  # unique, order-preserved
    )

    return df, shap_values
