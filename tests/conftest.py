"""Shared pytest fixtures using DrugSight schemas."""

import json
from pathlib import Path

import pandas as pd
import pytest

from drugsight.schemas import (
    BATCH_DOCKING_COLUMNS,
    DRUG_LIBRARY_COLUMNS,
    SCORING_FEATURES,
    TARGET_COLUMNS,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture()
def sample_targets_df() -> pd.DataFrame:
    """Load Huntington's targets as the default test fixture.

    The targets JSON is a dict keyed by disease_id. We extract the
    Huntington's disease entry (MONDO_0007739) for backwards compatibility
    with existing tests.
    """
    path = DATA_DIR / "sample_targets.json"
    with open(path) as f:
        data = json.load(f)
    # Support both dict-keyed and flat-list formats.
    if isinstance(data, dict):
        targets_list = data.get("MONDO_0007739", [])
    else:
        targets_list = data
    df = pd.DataFrame(targets_list)
    assert list(df.columns) == TARGET_COLUMNS
    return df


@pytest.fixture()
def sample_drug_library_df() -> pd.DataFrame:
    path = DATA_DIR / "sample_drugbank.csv"
    df = pd.read_csv(path)
    return df


@pytest.fixture()
def sample_docking_results_df() -> pd.DataFrame:
    path = DATA_DIR / "sample_docking_results.csv"
    df = pd.read_csv(path)
    for col in BATCH_DOCKING_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"
    return df


@pytest.fixture()
def sample_training_df() -> pd.DataFrame:
    path = DATA_DIR / "training_repurposing_cases.csv"
    df = pd.read_csv(path)
    for feat in SCORING_FEATURES:
        assert feat in df.columns, f"Missing feature: {feat}"
    assert "disease_id" in df.columns
    assert "repurposing_success" in df.columns
    return df
