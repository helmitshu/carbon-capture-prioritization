"""Tests for the prediction API (v2, backlog item 2)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from features import FEATURES

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v2"
    assert body["features"] == FEATURES


def test_predict_matches_local_model():
    bundle = joblib.load("model/registry/model_v2.pkl")
    tree = bundle["model"]
    X = pd.read_csv("data/facility_labeled.csv")[FEATURES].to_numpy()
    expected = tree.predict(X).tolist()
    expected_leaves = tree.apply(X).astype(int).tolist()

    r = client.post("/predict", json={"instances": X.tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"] == expected
    assert body["leaf_ids"] == expected_leaves
    assert body["model_version"] == "v2"


def test_predict_rejects_bad_shapes():
    assert client.post("/predict", json={"instances": []}).status_code == 422
    bad = client.post("/predict", json={"instances": [[1.0, 2.0]]})
    assert bad.status_code == 422
    nan = client.post("/predict",
                      content='{"instances": [[NaN, 1.0, 2.0]]}',
                      headers={"Content-Type": "application/json"})
    assert nan.status_code == 422


def test_structure_covers_walkthrough_needs():
    r = client.get("/model/structure")
    assert r.status_code == 200
    body = r.json()
    for key in ("children_left", "children_right", "feature", "threshold",
                "encoder_classes", "features", "labels"):
        assert key in body, key
    assert len(body["children_left"]) == body["n_nodes"]


def test_remote_client_matches_local():
    from model_client import get_tree_and_encoder, predict_batch
    import model_client
    # Point the client at the in-process TestClient server.
    orig_post, orig_get = model_client.requests.post, model_client.requests.get

    class _Resp:
        def __init__(self, payload, code=200):
            self._p, self.status_code = payload, code

        def json(self):
            return self._p

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    def fake_get(url, timeout=None):
        if url.endswith("/health"):
            return _Resp(client.get("/health").json())
        return _Resp(client.get("/model/structure").json())

    def fake_post(url, json=None, timeout=None):
        r = client.post("/predict", json=json)
        return _Resp(r.json(), r.status_code)

    model_client.requests.get, model_client.requests.post = fake_get, fake_post
    try:
        tree, le, source = get_tree_and_encoder()
        assert source == "api"
        fac = pd.read_csv("model/facilities.csv")
        preds, leaves = predict_batch(fac, tree)
        bundle = joblib.load("model/registry/model_v2.pkl")
        assert (preds == bundle["model"].predict(
            fac[FEATURES].to_numpy())).all()
        assert (leaves == bundle["model"].apply(
            fac[FEATURES].to_numpy())).all()
        assert list(le.classes_) == list(bundle["encoder"].classes_)
        assert (le.transform(["Conventional Oil and Gas Extraction"])
                == bundle["encoder"].transform(
                    ["Conventional Oil and Gas Extraction"])).all()
    finally:
        model_client.requests.get, model_client.requests.post = \
            orig_get, orig_post
