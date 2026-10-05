"""Shared feature engineering for the carbon-capture capstone pipeline.

Single source of truth for constants, feature derivation, and the model
configuration. scripts/03_features_model.py, scripts/04_save_model.py,
and app.py all import from here instead of re-deriving features
independently.
"""
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

# ---- constants (previously scattered across scripts and app.py) ----
TIER_THRESHOLD = 100_000   # tCO2e/yr, Alberta TIER regulation threshold
CO2_CUTOFF = 0.85          # label threshold: CCS vs CU candidate
FEATURES = ["log_emissions", "naics_sector_encoded", "years_reported"]
LABELS = ["CCS Candidate", "Potential CU Candidate"]
MODEL_PARAMS = dict(max_depth=3, min_samples_leaf=2, min_samples_split=6,
                    class_weight="balanced", random_state=42)
CV_PARAMS = dict(n_splits=5, shuffle=True, random_state=42)


def add_features(df, encoder=None, encoding="label", y=None,
                 smoothing=10.0):
    """Derive model features from a facility dataframe.

    Returns (X, encoder). Fits a new encoder unless one is supplied,
    in which case the supplied encoder is used for the transform.

    encoding="label" (default): v2 behavior, byte-identical. A plain
        sklearn LabelEncoder; arbitrary integers per sector.
    encoding="target": v3. A SectorEncoder(kind="target") mapping each
        sector to its smoothed candidate rate. Requires y as a binary
        0/1 array (1 = CCS Candidate). Chosen by scripts/08 on the
        national data (+0.057 macro F1, CU recall 0.51 -> 0.81).

    Gas-share columns are never used here: they define the label, so
    including them would leak.
    """
    df = df.copy()
    df["log_emissions"] = np.log1p(df["avg_annual_emissions"])
    sectors = df["sector"].astype(str)
    if encoder is None:
        if encoding == "label":
            encoder = LabelEncoder()
            df["naics_sector_encoded"] = encoder.fit_transform(sectors)
        elif encoding == "target":
            encoder = SectorEncoder(kind="target", smoothing=smoothing)
            df["naics_sector_encoded"] = encoder.fit(sectors, y).transform(
                sectors)
        else:
            raise ValueError(f"unknown encoding: {encoding!r}")
    else:
        df["naics_sector_encoded"] = encoder.transform(sectors)
    return df[FEATURES], encoder


class SectorEncoder:
    """Kind-aware sector encoder with a uniform interface.

    kind="label": wraps sklearn LabelEncoder (v2 semantics).
    kind="target": wraps category_encoders TargetEncoder (v3 semantics):
        each sector maps to its smoothed CCS-candidate rate. Unknown
        sectors at predict time fall back to the global mean instead of
        crashing (v2's LabelEncoder raised on unseen sectors).

    Uniform surface so app / api / client code works with either bundle:
        fit(sectors, y=None) -> self
        transform(sectors) -> 1-D float np.ndarray
        categories_ -> list of sectors seen in fit
        values() -> numeric value per category, aligned with categories_
        kind -> "label" | "target"
    """

    def __init__(self, kind="label", smoothing=10.0):
        if kind not in ("label", "target"):
            raise ValueError(f"unknown kind: {kind!r}")
        self.kind = kind
        self.smoothing = smoothing
        self.categories_ = []
        self._le = None
        self._te = None
        self.global_mean_ = float("nan")

    def fit(self, sectors, y=None):
        sectors = pd.Series(list(sectors)).astype(str)
        self.categories_ = sorted(sectors.unique())
        if self.kind == "label":
            self._le = LabelEncoder().fit(sectors)
        else:
            if y is None:
                raise ValueError(
                    "target encoding requires y (binary 0/1 array)")
            y_arr = np.asarray(y, dtype=float)
            if set(np.unique(y_arr)) - {0.0, 1.0}:
                raise ValueError("target encoding requires binary 0/1 y")
            self.global_mean_ = float(np.mean(y_arr))
            self._te = TargetEncoder(
                cols=["sector"], smoothing=self.smoothing,
                handle_unknown="value", handle_missing="value")
            self._te.fit(pd.DataFrame({"sector": sectors}),
                         pd.Series(y_arr))
        return self

    def transform(self, sectors):
        sectors = pd.Series(list(sectors)).astype(str)
        if self.kind == "label":
            return self._le.transform(sectors).astype(float)
        return self._te.transform(
            pd.DataFrame({"sector": sectors})).iloc[:, 0].to_numpy(
                dtype=float)

    def values(self):
        """Numeric value per category, aligned with categories_."""
        return self.transform(self.categories_)


def encoder_kind(enc):
    """'label' for v2 bundles (raw LabelEncoder), 'target' for v3."""
    return getattr(enc, "kind", "label")


def encoder_categories(enc):
    """Sector names known to the encoder, as strings."""
    if hasattr(enc, "categories_"):
        return [str(c) for c in enc.categories_]
    return [str(c) for c in enc.classes_]


def encoder_values(enc):
    """Numeric value per category, aligned with encoder_categories(enc)."""
    if hasattr(enc, "values"):
        return np.asarray(enc.values(), dtype=float)
    return np.asarray(enc.transform(enc.classes_), dtype=float)


def make_labels(df):
    """Apply the CO2-share labeling rule. Returns a copy with 'priority'."""
    df = df.copy()
    df["priority"] = np.where(df["co2_share"] >= CO2_CUTOFF, LABELS[0], LABELS[1])
    return df
