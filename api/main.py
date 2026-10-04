"""Prediction API (v2, backlog item 2).

Serves the registered capstone model over HTTP so the Streamlit app (and
any other client) gets predictions through JSON requests instead of
loading the model file in-process. This is the deployment pattern from
the MLOps course: the application talks to a deployed endpoint.

Endpoints:
  GET  /health           -> {"status", "model_version", "features"}
  POST /predict          -> {"instances": [[f1, f2, f3], ...]}
                            {"predictions": [...], "leaf_ids": [...],
                             "model_version": ...}
  GET  /model/structure  -> tree structure + encoder classes, so remote
                            clients can render walkthroughs without the
                            model file.

The served model version comes from MODEL_VERSION (default "v2") and is
loaded from the v2 model registry. Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = PROJECT_ROOT / "model" / "registry"

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v2")

app = FastAPI(title="Capstone prediction API")


class PredictRequest(BaseModel):
    instances: list[list[float]] = Field(
        ..., description="Rows of [log_emissions, naics_sector_encoded, "
                         "years_reported]")


class PredictResponse(BaseModel):
    predictions: list[str]
    leaf_ids: list[int]
    model_version: str


def _load_bundle():
    artifact = REGISTRY_DIR / f"model_{MODEL_VERSION}.pkl"
    card_path = REGISTRY_DIR / f"model_{MODEL_VERSION}.card.json"
    if not artifact.exists():
        raise RuntimeError(f"registry artifact missing: {artifact}")
    bundle = joblib.load(artifact)
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    return bundle, card


_bundle, _card = _load_bundle()
_tree = _bundle["model"]
_features: list[str] = list(_bundle["features"])
_labels: list[str] = list(_bundle["labels"])
_encoder_classes: list[str] = [str(c) for c in _bundle["encoder"].classes_]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION,
            "features": _features}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not req.instances:
        raise HTTPException(status_code=422, detail="instances is empty")
    n_expected = len(_features)
    for i, row in enumerate(req.instances):
        if len(row) != n_expected:
            raise HTTPException(
                status_code=422,
                detail=f"row {i}: expected {n_expected} features, "
                       f"got {len(row)}")
    X = np.array(req.instances, dtype=float)
    if not np.isfinite(X).all():
        raise HTTPException(status_code=422,
                            detail="instances contain NaN or inf")
    Xf = pd.DataFrame(X, columns=_features)
    preds = _tree.predict(Xf).tolist()
    leaves = _tree.apply(Xf).astype(int).tolist()
    return PredictResponse(predictions=preds, leaf_ids=leaves,
                           model_version=MODEL_VERSION)


@app.get("/model/structure")
def model_structure() -> dict:
    t = _tree.tree_
    return {
        "model_version": MODEL_VERSION,
        "features": _features,
        "labels": _labels,
        "encoder_classes": _encoder_classes,
        "n_nodes": int(t.node_count),
        "max_depth": int(_tree.max_depth),
        "children_left": t.children_left.astype(int).tolist(),
        "children_right": t.children_right.astype(int).tolist(),
        "feature": t.feature.astype(int).tolist(),
        "threshold": t.threshold.astype(float).tolist(),
    }
