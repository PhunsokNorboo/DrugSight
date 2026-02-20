"""
DrugSight End-to-End Pipeline Orchestrator.

Chains all six modules into a single ``run_pipeline()`` call that goes from
disease ID to ranked drug-repurposing candidates.  A companion
``run_pipeline_demo()`` function demonstrates the scoring and ranking stages
using pre-computed sample data so that no external APIs, AutoDock Vina, or
RDKit are required.

Usage
-----
Full pipeline (requires API access, Vina, and RDKit)::

    from drugsight.pipeline import run_pipeline
    ranked = run_pipeline("EFO_0000337", "data/drugbank.csv")

Quick demo (sample data only)::

    from drugsight.pipeline import run_pipeline_demo
    ranked = run_pipeline_demo()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from drugsight.config import DATA_DIR, RESULTS_DIR
from drugsight.schemas import RANKED_RESULT_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    disease_id: str,
    drugbank_csv: str | Path,
    training_csv: str | Path | None = None,
    min_association_score: float = 0.5,
    min_plddt: float = 70.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """Run the complete DrugSight pipeline from disease ID to ranked candidates.

    Parameters
    ----------
    disease_id:
        An EFO identifier (e.g. ``"EFO_0000337"`` for Huntington disease).
    drugbank_csv:
        Path to the DrugBank CSV file used by the drug-library module.
    training_csv:
        Path to a training CSV for the ML scorer.  When ``None``, defaults
        to ``data/training_repurposing_cases.csv``.
    min_association_score:
        Minimum Open Targets association score for a target to be included.
    min_plddt:
        Minimum average AlphaFold pLDDT score for a structure to be used.
    top_n:
        Number of top-ranked candidates to return and explain.

    Returns
    -------
    pd.DataFrame
        A DataFrame of the top *top_n* ranked candidates with columns
        matching ``RANKED_RESULT_COLUMNS``.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1/6: Disease targets ─────────────────────────────────────
    logger.info("Step 1/6: Finding disease targets...")
    try:
        from drugsight import disease_targets

        targets_df = disease_targets.get_disease_targets(
            disease_id, min_score=min_association_score
        )
        targets_df.to_csv(RESULTS_DIR / "targets.csv", index=False)
        logger.info(
            "Found %d target(s) for %s", len(targets_df), disease_id
        )
    except Exception as exc:
        logger.error("Step 1 failed — could not retrieve disease targets: %s", exc)
        raise

    # ── Step 2/6: AlphaFold structures ────────────────────────────────
    logger.info("Step 2/6: Retrieving AlphaFold structures...")
    try:
        from drugsight import alphafold_client

        uniprot_ids = targets_df["uniprot_id"].dropna().tolist()
        # Filter out empty strings that may remain after ID mapping.
        uniprot_ids = [uid for uid in uniprot_ids if uid]

        if not uniprot_ids:
            raise ValueError(
                "No UniProt IDs available — cannot fetch AlphaFold structures"
            )

        pdb_paths, confidences = alphafold_client.fetch_structures_batch(
            uniprot_ids, min_plddt=min_plddt
        )
        logger.info(
            "Retrieved %d structure(s) passing pLDDT >= %.1f",
            len(pdb_paths),
            min_plddt,
        )
    except Exception as exc:
        logger.error(
            "Step 2 failed — could not retrieve AlphaFold structures: %s", exc
        )
        raise

    if not pdb_paths:
        raise RuntimeError(
            "No AlphaFold structures passed the confidence filter "
            f"(min_plddt={min_plddt}).  Cannot proceed with docking."
        )

    # ── Step 3/6: Drug library ────────────────────────────────────────
    logger.info("Step 3/6: Preparing drug library...")
    try:
        from drugsight import drug_library

        drug_df = drug_library.prepare_library(drugbank_csv)
        logger.info("Drug library contains %d compounds", len(drug_df))
    except Exception as exc:
        logger.error("Step 3 failed — could not prepare drug library: %s", exc)
        raise

    # ── Step 4/6: Molecular docking ───────────────────────────────────
    logger.info("Step 4/6: Running molecular docking...")
    try:
        from drugsight import docking_engine

        # Build a quick lookup from uniprot_id to target row.
        target_lookup = {
            row["uniprot_id"]: row
            for _, row in targets_df.iterrows()
            if row["uniprot_id"]
        }

        docking_dfs: list[pd.DataFrame] = []
        for pdb_path, confidence in zip(pdb_paths, confidences):
            uid = confidence["uniprot_id"]
            target_row = target_lookup.get(uid)
            if target_row is None:
                logger.warning(
                    "No matching target row for UniProt %s — skipping docking",
                    uid,
                )
                continue

            target_symbol = target_row["symbol"]
            logger.info(
                "Docking library against %s (%s)", target_symbol, uid
            )
            docking_df = docking_engine.batch_dock(
                pdb_path,
                drug_df,
                uid,
                target_symbol,
            )
            docking_dfs.append(docking_df)

        if not docking_dfs:
            raise RuntimeError("All docking runs failed or were skipped")

        all_docking = pd.concat(docking_dfs, ignore_index=True)
        all_docking.to_csv(RESULTS_DIR / "docking_results.csv", index=False)
        logger.info("Docking produced %d result rows", len(all_docking))

    except Exception as exc:
        logger.error("Step 4 failed — molecular docking error: %s", exc)
        raise

    # ── Step 5/6: ML scoring ──────────────────────────────────────────
    logger.info("Step 5/6: Training ML model and scoring...")
    try:
        from drugsight import ml_scorer

        resolved_training_csv = (
            Path(training_csv)
            if training_csv is not None
            else DATA_DIR / "training_repurposing_cases.csv"
        )

        model = ml_scorer.train_ranker(resolved_training_csv)
        feature_df = ml_scorer.merge_features(
            all_docking, confidences, targets_df
        )
        scored_df = ml_scorer.score_candidates(feature_df, model)
        ranked_df, shap_values = ml_scorer.explain_top_candidates(
            scored_df, model, top_n
        )
        logger.info(
            "Scoring complete — %d candidates ranked, top %d explained",
            len(ranked_df),
            top_n,
        )
    except Exception as exc:
        logger.error("Step 5 failed — ML scoring error: %s", exc)
        raise

    # ── Step 6/6: Save results ────────────────────────────────────────
    logger.info("Step 6/6: Saving results...")
    try:
        result = ranked_df.head(top_n)
        result.to_csv(RESULTS_DIR / "ranked_candidates.csv", index=False)
        logger.info(
            "Saved top %d candidates to %s",
            len(result),
            RESULTS_DIR / "ranked_candidates.csv",
        )
    except Exception as exc:
        logger.error("Step 6 failed — could not save results: %s", exc)
        raise

    return result


# ---------------------------------------------------------------------------
# Demo pipeline (pre-computed sample data, no external dependencies)
# ---------------------------------------------------------------------------


def run_pipeline_demo(
    disease_id: str = "EFO_0000337",
) -> pd.DataFrame:
    """Run a quick demo of the scoring pipeline using pre-computed sample data.

    This function requires no external API access, no AutoDock Vina, and no
    RDKit installation.  It loads sample docking results and training data
    from the ``data/`` directory, trains (or falls back to a weighted-sum)
    model, merges features, scores, and ranks the candidates.

    Parameters
    ----------
    disease_id:
        EFO identifier used to label the demo run.  Defaults to
        ``"EFO_0000337"`` (Huntington disease).

    Returns
    -------
    pd.DataFrame
        Top 20 ranked candidates with explanations.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Running DrugSight demo pipeline for %s (pre-computed data)", disease_id
    )

    # ── Load pre-computed sample data ─────────────────────────────────
    sample_docking_path = DATA_DIR / "sample_docking_results.csv"
    training_path = DATA_DIR / "training_repurposing_cases.csv"
    sample_targets_path = DATA_DIR / "sample_targets.json"

    if not sample_docking_path.exists():
        raise FileNotFoundError(
            f"Sample docking results not found at {sample_docking_path}. "
            "Ensure the data/ directory contains the pre-computed sample files."
        )

    logger.info("Loading sample docking results from %s", sample_docking_path)
    docking_df = pd.read_csv(sample_docking_path)
    logger.info("Loaded %d docking rows", len(docking_df))

    # ── Load sample targets for association scores ────────────────────
    if sample_targets_path.exists():
        logger.info("Loading sample targets from %s", sample_targets_path)
        with open(sample_targets_path, "r") as fh:
            targets_data = json.load(fh)

        # Support both dict-keyed-by-disease and flat-list formats.
        if isinstance(targets_data, dict):
            disease_targets = targets_data.get(disease_id, [])
            targets_df = pd.DataFrame(disease_targets)
        else:
            targets_df = pd.DataFrame(targets_data)

        logger.info("Loaded %d sample targets for %s", len(targets_df), disease_id)
    else:
        logger.warning(
            "Sample targets JSON not found at %s — association scores will "
            "use defaults",
            sample_targets_path,
        )
        targets_df = pd.DataFrame()

    # ── Filter docking results to this disease's targets ────────────────
    if not targets_df.empty and "uniprot_id" in targets_df.columns:
        disease_uniprots = set(targets_df["uniprot_id"].dropna().tolist())
        if disease_uniprots:
            docking_df = docking_df[
                docking_df["uniprot_id"].isin(disease_uniprots)
            ].copy()
            logger.info(
                "Filtered docking to %d rows for %d disease targets",
                len(docking_df),
                len(disease_uniprots),
            )

    # ── Build synthetic AlphaFold confidence entries ───────────────────
    # The demo has no real AlphaFold calls, so we synthesize confidence
    # records from the UniProt IDs present in the docking results.
    unique_uniprots = docking_df["uniprot_id"].unique().tolist()
    confidences = [
        {
            "uniprot_id": uid,
            "avg_plddt": 85.0,  # reasonable default for demo
            "model_url": f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb",
            "version": 4,
        }
        for uid in unique_uniprots
    ]
    logger.info(
        "Synthesized %d confidence records for demo", len(confidences)
    )

    # ── Train model and score ─────────────────────────────────────────
    try:
        from drugsight import ml_scorer

        logger.info("Training model on %s", training_path)
        model = ml_scorer.train_ranker(training_path)

        logger.info("Merging features...")
        feature_df = ml_scorer.merge_features(
            docking_df, confidences, targets_df
        )

        logger.info("Scoring candidates...")
        scored_df = ml_scorer.score_candidates(feature_df, model)

        top_n = 20
        logger.info("Explaining top %d candidates...", top_n)
        ranked_df, shap_values = ml_scorer.explain_top_candidates(
            scored_df, model, top_n
        )
    except Exception as exc:
        logger.error("Demo scoring failed: %s", exc)
        raise

    # ── Save demo results ─────────────────────────────────────────────
    result = ranked_df.head(20)
    output_path = RESULTS_DIR / "ranked_candidates.csv"
    result.to_csv(output_path, index=False)
    logger.info("Demo complete — saved %d candidates to %s", len(result), output_path)

    return result
