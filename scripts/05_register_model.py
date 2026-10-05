"""Phase 5b: register the trained model in the v2 model registry.

Registers an EXISTING trained model artifact (model/model.pkl) as a
versioned registry entry: a versioned artifact copy, a model card with
full lineage (params, metrics, data hashes), and a registry index that
tracks each version's lifecycle stage.

This is the "store and register before deployment" step from the MLOps
course: no model reaches production without a registry record that ties
it to its training data, metrics, and code.

Usage:
    .venv/bin/python scripts/05_register_model.py [--version v1] [--stage production]
                                                 [--training-date 2026-10-03]
                                                 [--source-artifact model/model.pkl]
                                                 [--source-data data/facility_labeled.csv]
                                                 [--training-script scripts/04_save_model.py]
                                                 [--metrics-json hidden_files/v3_metrics.json]

Idempotent: re-running with the same version overwrites the card and
re-verifies the artifact.

The --source-*, --training-script and --metrics-json flags exist for v3:
they point registration at a non-default artifact/data/script and let the
training script supply its own leakage-safe CV metrics (the default
evaluate_protocol refits the tree on the encoded matrix, which is only
valid when the encoding itself uses no target information).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES, LABELS, MODEL_PARAMS, CV_PARAMS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
REGISTRY_DIR = MODEL_DIR / "registry"
REGISTRY_INDEX = REGISTRY_DIR / "registry.json"

# The artifact this version registers. v1 registers the model that is
# already trained and live in the v1 app: same weights, now tracked.
SOURCE_ARTIFACT = MODEL_DIR / "model.pkl"
SOURCE_DATA = PROJECT_ROOT / "data" / "facility_labeled.csv"
RAW_DATA = PROJECT_ROOT / "data" / "Capstone_Dataset_clean.csv"


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> dict:
    """Load the registry index, or start empty."""
    if REGISTRY_INDEX.exists():
        return json.loads(REGISTRY_INDEX.read_text())
    return {"versions": {}}


def evaluate_protocol(X: pd.DataFrame, y: pd.Series) -> dict:
    """Recompute 5-fold stratified CV metrics for the model configuration.

    This measures the training protocol (MODEL_PARAMS / CV_PARAMS), not
    the registered artifact itself: the artifact is fit on the full
    dataset, so cross-validated metrics can only come from refitting.
    The card records the protocol explicitly so nobody mistakes these
    for holdout metrics of the artifact.
    """
    tree = DecisionTreeClassifier(**MODEL_PARAMS)
    cv = StratifiedKFold(**CV_PARAMS)
    y_pred = cross_val_predict(tree, X, y, cv=cv)
    report = classification_report(
        y, y_pred, labels=LABELS, digits=4, output_dict=True
    )
    return {
        "protocol": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), "
                    "cross_val_predict",
        "accuracy": round(report["accuracy"], 4),
        "macro_avg": {
            k: round(v, 4)
            for k, v in report["macro avg"].items()
            if k in ("precision", "recall", "f1-score")
        },
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
    }


def build_card(version: str, stage: str, training_date: str,
               source_data: Path = SOURCE_DATA,
               training_script: str = "scripts/04_save_model.py",
               metrics_json: Path | None = None) -> dict:
    """Assemble the model card dict for this version."""
    above = pd.read_csv(source_data)
    X = above[FEATURES]
    y = above["priority"]
    # v3 was cleaned from the national raw extract; v1/v2 from Alberta.
    clean_name = ("data/Capstone_Dataset_clean_national.csv"
                  if "national" in source_data.name
                  else "data/Capstone_Dataset_clean.csv")
    clean_path = PROJECT_ROOT / clean_name
    if metrics_json is not None:
        supplied = json.loads(Path(metrics_json).read_text())
        metrics = {
            "protocol": supplied["protocol"],
            "accuracy": supplied["accuracy"],
            "macro_avg": supplied["macro_avg"],
            "per_class": supplied["per_class"],
        }
        limitations = supplied.get("limitations", [])
    else:
        metrics = evaluate_protocol(X, y)
        limitations = [
            "CU class precision is low (~0.30): most 'Potential CU' "
            "predictions are actually CCS. See the disagreement banner.",
            "Model errors concentrate in smaller true-CCS facilities.",
            "Single decision tree chosen for interpretability, not "
            "maximum performance.",
        ]

    return {
        "version": version,
        "stage": stage,
        "model_type": "DecisionTreeClassifier (scikit-learn)",
        "params": MODEL_PARAMS,
        "features": FEATURES,
        "labels": LABELS,
        "label_rule": "co2_share >= 0.85 -> CCS Candidate else Potential CU Candidate "
                      "(gas shares excluded from features: label leakage)",
        "metrics": metrics,
        "data": {
            "training_file": str(source_data.relative_to(PROJECT_ROOT))
                             if source_data.is_relative_to(PROJECT_ROOT)
                             else source_data.name,
            "training_sha256": sha256_of(source_data),
            "source_clean_file": clean_name,
            "source_clean_sha256": sha256_of(clean_path),
            "n_facilities": int(len(above)),
            "class_balance": {
                label: int((y == label).sum()) for label in LABELS
            },
        },
        "artifact": {
            "file": f"registry/model_{version}.pkl",
            "sha256": "",  # filled after the artifact copy is written
        },
        "training_script": training_script,
        "training_date": training_date,
        "registered_date": date.today().isoformat(),
        "lesson_spec": "AMII AI Pathways Technical Track, Capstone Project",
        "limitations": limitations,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: temp file + rename, so a crash can never
    leave a half-written registry index behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def register(version: str, stage: str, training_date: str,
             source_artifact: Path = SOURCE_ARTIFACT,
             source_data: Path = SOURCE_DATA,
             training_script: str = "scripts/04_save_model.py",
             metrics_json: Path | None = None) -> Path:
    """Register source_artifact as `version`; return the card path."""
    if not source_artifact.exists():
        raise FileNotFoundError(f"missing source artifact: {source_artifact}")
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(source_artifact)

    # 1. Versioned artifact copy.
    artifact_path = REGISTRY_DIR / f"model_{version}.pkl"
    joblib.dump(bundle, artifact_path)

    # 2. Verify the copy: reload and confirm identical predictions.
    reloaded = joblib.load(artifact_path)
    above = pd.read_csv(source_data)
    X = above[FEATURES].to_numpy()
    original_pred = bundle["model"].predict(X)
    reloaded_pred = reloaded["model"].predict(X)
    if not np.array_equal(original_pred, reloaded_pred):
        raise RuntimeError("registry artifact verification failed: "
                           "predictions differ from source")
    if list(bundle["features"]) != FEATURES:
        raise RuntimeError("registry artifact verification failed: "
                           "feature list mismatch")

    # 3. Model card.
    card = build_card(version, stage, training_date,
                      source_data=source_data,
                      training_script=training_script,
                      metrics_json=metrics_json)
    card["artifact"]["sha256"] = sha256_of(artifact_path)
    card_path = REGISTRY_DIR / f"model_{version}.card.json"
    card_path.write_text(json.dumps(card, indent=2) + "\n")

    # 4. Registry index (atomic write).
    registry = load_registry()
    registry["versions"][version] = {
        "stage": stage,
        "card": card_path.name,
        "artifact": artifact_path.name,
        "artifact_sha256": card["artifact"]["sha256"],
        "registered_date": card["registered_date"],
    }
    _atomic_write_json(REGISTRY_INDEX, registry)

    return card_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register the trained model in the v2 model registry.")
    parser.add_argument("--version", default="v1",
                        help="registry version label (default: v1)")
    parser.add_argument("--stage", default="production",
                        choices=["staging", "production", "archived"],
                        help="lifecycle stage (default: production)")
    parser.add_argument("--training-date", default=None,
                        help="training date YYYY-MM-DD (default: the source "
                             "artifact file's modification date)")
    parser.add_argument("--source-artifact", default=str(SOURCE_ARTIFACT),
                        help="model artifact to register "
                             "(default: model/model.pkl)")
    parser.add_argument("--source-data", default=str(SOURCE_DATA),
                        help="labeled training CSV "
                             "(default: data/facility_labeled.csv)")
    parser.add_argument("--training-script",
                        default="scripts/04_save_model.py",
                        help="script that trained the artifact, for the card")
    parser.add_argument("--metrics-json", default=None,
                        help="JSON with leakage-safe CV metrics "
                             "{protocol, accuracy, macro_avg, per_class, "
                             "limitations} to use instead of recomputing")
    args = parser.parse_args()

    source_artifact = Path(args.source_artifact)
    training_date = args.training_date
    if training_date is None:
        training_date = date.fromtimestamp(
            source_artifact.stat().st_mtime).isoformat()

    card_path = register(
        args.version, args.stage, training_date,
        source_artifact=source_artifact,
        source_data=Path(args.source_data),
        training_script=args.training_script,
        metrics_json=Path(args.metrics_json)
        if args.metrics_json else None)
    card = json.loads(card_path.read_text())
    print(f"registered model_{args.version}.pkl "
          f"(stage={args.stage}, artifact verified)")
    print(f"card: {card_path}")
    m = card["metrics"]
    print(f"CV accuracy={m['accuracy']}, "
          f"CCS F1={m['per_class']['CCS Candidate']['f1']}, "
          f"CU recall={m['per_class']['Potential CU Candidate']['recall']}")


if __name__ == "__main__":
    main()
