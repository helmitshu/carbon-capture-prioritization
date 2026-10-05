"""Tests for v3 (national model): encoder, bundle, tiers, registry.

v2 behavior is covered by the existing suite; these tests pin the v3
additions without touching v2 expectations.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from features import (SectorEncoder, add_features, encoder_categories,
                      encoder_kind, encoder_values, FEATURES, LABELS)


@pytest.fixture(scope="module")
def alberta():
    return pd.read_csv("data/facility_labeled.csv")


@pytest.fixture(scope="module")
def national():
    return pd.read_csv("data/facility_labeled_national.csv")


@pytest.fixture(scope="module")
def v3_bundle():
    return joblib.load("model/registry/model_v3.pkl")


@pytest.fixture(scope="module")
def v3_facilities():
    return pd.read_csv("model/facilities_v3.csv")


# ---- SectorEncoder ----

def test_label_kind_matches_sklearn(alberta):
    enc = SectorEncoder(kind="label").fit(alberta["sector"])
    ref = LabelEncoder().fit(alberta["sector"].astype(str))
    assert list(enc.categories_) == [str(c) for c in ref.classes_]
    assert np.array_equal(enc.transform(alberta["sector"]),
                          ref.transform(
                              alberta["sector"].astype(str)).astype(float))


def test_target_kind_bounded_and_deterministic(national):
    y = (national["priority"] == LABELS[0]).astype(int).to_numpy()
    e1 = SectorEncoder(kind="target", smoothing=10.0).fit(national["sector"], y)
    e2 = SectorEncoder(kind="target", smoothing=10.0).fit(national["sector"], y)
    v = e1.values()
    assert ((v >= 0) & (v <= 1)).all(), "target means must be in [0,1]"
    assert np.array_equal(e1.transform(national["sector"]),
                          e2.transform(national["sector"]))


def test_target_unknown_sector_falls_back_to_global_mean(national):
    y = (national["priority"] == LABELS[0]).astype(int).to_numpy()
    enc = SectorEncoder(kind="target").fit(national["sector"], y)
    out = enc.transform(pd.Series(["No Such Sector"]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(enc.global_mean_)


def test_target_requires_binary_y(national):
    with pytest.raises(ValueError):
        SectorEncoder(kind="target").fit(national["sector"])
    with pytest.raises(ValueError):
        SectorEncoder(kind="target").fit(national["sector"],
                                         national["priority"])
    with pytest.raises(ValueError):
        SectorEncoder(kind="mystery")


def test_helpers_accept_raw_label_encoder(alberta):
    # v2 bundles store a raw LabelEncoder: helpers must not break on it.
    le = LabelEncoder().fit(alberta["sector"].astype(str))
    assert encoder_kind(le) == "label"
    assert encoder_categories(le) == [str(c) for c in le.classes_]
    assert np.array_equal(encoder_values(le),
                          le.transform(le.classes_).astype(float))


def test_add_features_default_is_v2_identical(alberta):
    X, enc = add_features(alberta)
    assert isinstance(enc, LabelEncoder)
    ref = LabelEncoder().fit_transform(alberta["sector"].astype(str))
    assert np.array_equal(X["naics_sector_encoded"].to_numpy(), ref)
    assert list(X.columns) == FEATURES


# ---- v3 bundle ----

def test_v3_bundle_contract(v3_bundle):
    assert v3_bundle["encoder"].kind == "target"
    assert list(v3_bundle["features"]) == FEATURES
    assert list(v3_bundle["labels"]) == LABELS
    assert encoder_categories(v3_bundle["encoder"])


def test_v3_bundle_predicts_on_serving_table(v3_bundle, v3_facilities):
    X = v3_facilities[FEATURES].to_numpy()
    pred = v3_bundle["model"].predict(X)
    assert len(pred) == len(v3_facilities)
    assert set(pred) <= set(LABELS)


def test_v3_serving_column_matches_encoder(v3_bundle, v3_facilities):
    enc = v3_bundle["encoder"]
    expected = enc.transform(v3_facilities["sector"])
    assert np.allclose(v3_facilities["naics_sector_encoded"].to_numpy(),
                       expected), "serving table must carry target-encoded values"


# ---- tiers and reasons ----

def test_v3_tiers_valid(v3_facilities):
    assert set(v3_facilities["verdict_tier"].unique()) <= {
        "unanimous", "majority", "contested"}
    assert v3_facilities["panel_ccs_votes"].between(0, 4).all()
    # tier consistent with votes
    v = v3_facilities["panel_ccs_votes"]
    t = v3_facilities["verdict_tier"]
    assert ((v.isin([0, 4])) == (t == "unanimous")).all()
    assert ((v == 2) == (t == "contested")).all()


def test_v3_reasons_present(v3_facilities):
    assert (v3_facilities["reason_1"].astype(str) != "").all()
    # reasons mention the feature that split
    sample = v3_facilities["reason_1"].iloc[0]
    assert any(k in sample for k in ("Scale:", "Sector ", "Reporting history:"))


# ---- registry ----

def test_registry_v3_staging_v2_still_production():
    reg = json.load(open("model/registry/registry.json"))
    assert reg["versions"]["v3"]["stage"] == "staging"
    assert reg["versions"]["v2"]["stage"] == "production"
    assert reg["production"] == "v2"
    card = json.load(open("model/registry/model_v3.card.json"))
    assert card["data"]["n_facilities"] == 454
    assert card["metrics"]["accuracy"] >= 0.75
