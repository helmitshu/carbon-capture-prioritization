"""Phase 5c: model validation gate.

Automated checks that run BEFORE a registry version may be promoted to
production. Any failure exits non-zero with a clear report, which is
what fails the deploy in a pipeline.

Checks, in order:
  1. Artifact integrity: the registry artifact's SHA-256 matches the card.
  2. Acceptance bars: 5-fold stratified CV recomputed here must clear
     accuracy >= 0.70, CCS Candidate F1 >= 0.80,
     Potential CU Candidate recall >= 0.50 (bars from ACCEPTANCE_CRITERIA.md,
     overridable via CLI).
  3. Input schema: the artifact's feature list matches the expected
     3-feature schema; a synthetic valid row predicts without error.
  4. Sanity cases: the 3 largest emitters (unambiguous CCS cases by the
     85% CO2-share label rule) must predict "CCS Candidate"; every one
     of the 150 facilities must get a non-null prediction in a known label.

Usage:
    .venv/bin/python scripts/06_validate_model.py --version v2
    .venv/bin/python scripts/06_validate_model.py --version v2 \
        --min-accuracy 0.70 --min-ccs-f1 0.80 --min-cu-recall 0.50

Exit code 0 = PASS (safe to promote). Non-zero = FAIL (do not deploy).
Writes hidden_files/validation_<version>.json with the full report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
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
REGISTRY_DIR = PROJECT_ROOT / "model" / "registry"
DATA = PROJECT_ROOT / "data" / "facility_labeled.csv"
HIDDEN = PROJECT_ROOT / "hidden_files"

EXPECTED_SCHEMA = FEATURES  # ["log_emissions", "naics_sector_encoded", "years_reported"]

# Acceptance bars, the single source of truth. scripts/07_promote_model.py
# imports these (via the gate module) instead of hardcoding its own copy,
# so a bar change in one place applies everywhere. The human-readable
# version lives in ACCEPTANCE_CRITERIA.md; these numbers must match it.
DEFAULT_BARS = {
    "min_accuracy": 0.70,
    "min_ccs_f1": 0.80,
    "min_cu_recall": 0.50,
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GateReport:
    version: str
    passed: bool = True
    checks: list = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name, passed, detail))
        if not passed:
            self.passed = False

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_artifact_integrity(version: str, report: GateReport) -> dict | None:
    """Check 1: artifact hash matches the registry card. Returns the card."""
    card_path = REGISTRY_DIR / f"model_{version}.card.json"
    artifact_path = REGISTRY_DIR / f"model_{version}.pkl"
    if not card_path.exists():
        report.add("artifact_integrity", False,
                   f"missing card: {card_path.name}")
        return None
    if not artifact_path.exists():
        report.add("artifact_integrity", False,
                   f"missing artifact: {artifact_path.name}")
        return None
    card = json.loads(card_path.read_text())
    actual = sha256_of(artifact_path)
    expected = card["artifact"]["sha256"]
    report.add("artifact_integrity", actual == expected,
               f"sha256 {'matches' if actual == expected else 'MISMATCH'}")
    if actual != expected:
        return None
    # The model must be validated against the data it was trained on.
    data_actual = sha256_of(DATA)
    data_expected = card["data"]["training_sha256"]
    report.add("training_data_lineage", data_actual == data_expected,
               "training data matches card"
               if data_actual == data_expected
               else "MISMATCH: data changed since training; "
                    "re-register before validating")
    if data_actual != data_expected:
        return None
    return card


def check_acceptance_bars(version: str, bars: dict,
                          report: GateReport) -> None:
    """Check 2: recomputed 5-fold CV metrics clear the acceptance bars."""
    above = pd.read_csv(DATA)
    X, y = above[FEATURES], above["priority"]
    tree = DecisionTreeClassifier(**MODEL_PARAMS)
    cv = StratifiedKFold(**CV_PARAMS)
    y_pred = cross_val_predict(tree, X, y, cv=cv)
    r = classification_report(y, y_pred, labels=LABELS, digits=4,
                              output_dict=True)
    metrics = {
        "accuracy": r["accuracy"],
        "ccs_f1": r["CCS Candidate"]["f1-score"],
        "cu_recall": r["Potential CU Candidate"]["recall"],
    }
    bar_map = {"accuracy": bars["min_accuracy"],
               "ccs_f1": bars["min_ccs_f1"],
               "cu_recall": bars["min_cu_recall"]}
    for key, value in metrics.items():
        ok = value >= bar_map[key]
        report.add(f"acceptance_{key}", ok,
                   f"{value:.4f} vs bar {bar_map[key]:.2f}")


def check_input_schema(version: str, report: GateReport) -> np.ndarray | None:
    """Check 3: feature schema matches; synthetic row predicts cleanly."""
    bundle = joblib.load(REGISTRY_DIR / f"model_{version}.pkl")
    model = bundle["model"]
    ok_names = list(bundle["features"]) == EXPECTED_SCHEMA
    report.add("schema_feature_names", ok_names,
               f"artifact features={list(bundle['features'])}")
    if not ok_names:
        return None
    try:
        sample = pd.DataFrame(
            [[12.5, 3, 10]], columns=EXPECTED_SCHEMA)
        pred = model.predict(sample)
        ok_pred = pred[0] in LABELS
        report.add("schema_synthetic_predict", ok_pred,
                   f"predicted {pred[0]}")
    except Exception as e:  # noqa: BLE001 - gate must report, not crash
        report.add("schema_synthetic_predict", False, f"raised: {e}")
        return None
    return model


def check_sanity_cases(model, report: GateReport) -> None:
    """Check 4: obvious cases predict sanely; full coverage, no nulls."""
    above = pd.read_csv(DATA)
    X_all = above[FEATURES].to_numpy()
    preds = model.predict(X_all)
    report.add("sanity_full_coverage",
               len(preds) == len(above) and
               set(pd.Series(preds).unique()) <= set(LABELS),
               f"{len(preds)} predictions, labels={sorted(set(preds))}")
    # The 3 largest emitters are unambiguous CCS cases by the label rule.
    top3 = above.nlargest(3, "avg_annual_emissions")
    top3_pred = model.predict(top3[FEATURES].to_numpy())
    ok = all(p == "CCS Candidate" for p in top3_pred)
    report.add("sanity_top_emitters_ccs", ok,
               f"top-3 emitters predicted {list(top3_pred)}")


def run_gate(version: str, bars: dict) -> GateReport:
    report = GateReport(version=version)
    card = check_artifact_integrity(version, report)
    if card is None:
        return report  # cannot trust anything else
    check_acceptance_bars(version, bars, report)
    model = check_input_schema(version, report)
    if model is not None:
        check_sanity_cases(model, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation gate: PASS/FAIL a registry version "
                    "before production promotion.")
    parser.add_argument("--version", required=True,
                        help="registry version to validate (e.g. v2)")
    parser.add_argument("--min-accuracy", type=float,
                        default=DEFAULT_BARS["min_accuracy"])
    parser.add_argument("--min-ccs-f1", type=float,
                        default=DEFAULT_BARS["min_ccs_f1"])
    parser.add_argument("--min-cu-recall", type=float,
                        default=DEFAULT_BARS["min_cu_recall"])
    args = parser.parse_args()
    bars = {"min_accuracy": args.min_accuracy,
            "min_ccs_f1": args.min_ccs_f1,
            "min_cu_recall": args.min_cu_recall}

    report = run_gate(args.version, bars)
    HIDDEN.mkdir(parents=True, exist_ok=True)
    out = HIDDEN / f"validation_{args.version}.json"
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    status = "PASS" if report.passed else "FAIL"
    print(f"validation gate {args.version}: {status}")
    for c in report.checks:
        mark = "ok" if c.passed else "FAILED"
        print(f"  [{mark}] {c.name}: {c.detail}")
    print(f"report: {out}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
