"""Model client for the capstone app (v2, backlog item 2).

The app gets its model through this module. It tries the prediction API
first (PREDICT_API_URL, default http://localhost:8000) and falls back to
loading the local model file if the API is unreachable, so the app never
goes down because the API does.

Both paths expose the same interface the app already uses:
  tree.predict(X)   -> array of labels
  tree.apply(X)     -> array of leaf ids
  tree.tree_        -> children_left / children_right / feature / threshold
  le.classes_       -> encoder classes
  le.transform(...) -> code lookup

Remote objects fetch from the API; local objects are the real sklearn
artifacts. Callers cannot tell the difference.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import requests

from features import FEATURES

API_URL = os.environ.get("PREDICT_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("PREDICT_API_TIMEOUT", "5"))
LOCAL_BUNDLE = "model/model.pkl"


class _RemoteTree:
    """sklearn-compatible tree backed by the prediction API."""

    def __init__(self, base_url: str, struct: dict) -> None:
        self._base = base_url
        self._version = struct["model_version"]
        self.tree_ = SimpleNamespace(
            node_count=struct["n_nodes"],
            children_left=np.array(struct["children_left"]),
            children_right=np.array(struct["children_right"]),
            feature=np.array(struct["feature"]),
            threshold=np.array(struct["threshold"]),
            n_node_samples=np.array(struct["n_node_samples"]),
        )
        self.max_depth = struct["max_depth"]

    def _post(self, X) -> dict:
        instances = np.asarray(X, dtype=float).tolist()
        r = requests.post(f"{self._base}/predict",
                          json={"instances": instances}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def predict(self, X) -> np.ndarray:
        return np.array(self._post(X)["predictions"])

    def apply(self, X) -> np.ndarray:
        return np.array(self._post(X)["leaf_ids"])


class _RemoteEncoder:
    """LabelEncoder-compatible shim backed by the API's class list."""

    def __init__(self, classes: list[str]) -> None:
        self.classes_ = np.array(classes)

    def transform(self, values) -> np.ndarray:
        idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([idx[str(v)] for v in np.ravel(values)])


def _from_api():
    """Return (tree, encoder, features, version) from the API.

    Raises on any failure so the caller can fall back to local.
    """
    health = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
    health.raise_for_status()
    info = health.json()
    if list(info["features"]) != FEATURES:
        raise ValueError(
            f"API feature schema {info['features']} != local {FEATURES}")
    struct = requests.get(f"{API_URL}/model/structure",
                          timeout=TIMEOUT).json()
    tree = _RemoteTree(API_URL, struct)
    le = _RemoteEncoder(struct["encoder_classes"])
    return tree, le, info["model_version"]


def get_tree_and_encoder():
    """Return (tree, encoder, source).

    source is "api" when the prediction API served the model, "local"
    when it fell back to the bundled model file.
    """
    try:
        tree, le, version = _from_api()
        print(f"model client: serving {version} via API ({API_URL})")
        return tree, le, "api"
    except Exception as e:  # noqa: BLE001 - any API failure -> fallback
        print(f"model client: API unreachable ({e}); using local model")
        bundle = joblib.load(LOCAL_BUNDLE)
        assert list(bundle["features"]) == FEATURES, "local schema drift"
        return bundle["model"], bundle["encoder"], "local"


def predict_batch(df: pd.DataFrame, tree) -> tuple[np.ndarray, np.ndarray]:
    """Predictions + leaf ids for a feature DataFrame via tree.predict/apply."""
    X = df[FEATURES].to_numpy(dtype=float)
    return tree.predict(X), tree.apply(X)
