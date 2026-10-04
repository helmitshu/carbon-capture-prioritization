"""Shared feature engineering for the carbon-capture capstone pipeline.

Single source of truth for constants, feature derivation, and the model
configuration. scripts/03_features_model.py, scripts/04_save_model.py,
and app.py all import from here instead of re-deriving features
independently.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---- constants (previously scattered across scripts and app.py) ----
TIER_THRESHOLD = 100_000   # tCO2e/yr, Alberta TIER regulation threshold
CO2_CUTOFF = 0.85          # label threshold: CCS vs CU candidate
FEATURES = ["log_emissions", "naics_sector_encoded", "years_reported"]
LABELS = ["CCS Candidate", "Potential CU Candidate"]
MODEL_PARAMS = dict(max_depth=3, min_samples_leaf=2, min_samples_split=6,
                    class_weight="balanced", random_state=42)
CV_PARAMS = dict(n_splits=5, shuffle=True, random_state=42)


def add_features(df, encoder=None):
    """Derive model features from a facility dataframe.

    Returns (X, encoder). Fits a new LabelEncoder unless one is supplied,
    in which case the supplied encoder is used for the transform.
    Gas-share columns are never used here: they define the label, so
    including them would leak.
    """
    df = df.copy()
    df["log_emissions"] = np.log1p(df["avg_annual_emissions"])
    if encoder is None:
        encoder = LabelEncoder()
        df["naics_sector_encoded"] = encoder.fit_transform(df["sector"].astype(str))
    else:
        df["naics_sector_encoded"] = encoder.transform(df["sector"].astype(str))
    return df[FEATURES], encoder


def make_labels(df):
    """Apply the CO2-share labeling rule. Returns a copy with 'priority'."""
    df = df.copy()
    df["priority"] = np.where(df["co2_share"] >= CO2_CUTOFF, LABELS[0], LABELS[1])
    return df
