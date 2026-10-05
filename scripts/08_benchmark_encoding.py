"""Benchmark: sector encoding strategies for the facility screening model.

Leakage-safe: every target-based encoder lives INSIDE a sklearn Pipeline
so it is fit on train folds only. The label baseline replicates
scripts/03 exactly (LabelEncoder is unsupervised, so fitting it once
upfront leaks nothing). Identical folds for every contender: same
StratifiedKFold(5, shuffle, rs=42) object, same row order, same tree
params. Differences smaller than ~0.03 macro F1 are fold noise on this
150-facility dataset, not signal.

Usage: python3 scripts/08_benchmark_encoding.py [--data path/to/facility_labeled.csv]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from features import (add_features, FEATURES, LABELS,
                      MODEL_PARAMS, CV_PARAMS)

# ---- facility-level labeled data (single canonical row order) ----
# scripts/03 (Alberta) and scripts/11 (national) produce these files
# with identical schemas; every contender below sees the same rows,
# so the folds are identical and the comparison is fair.
parser = argparse.ArgumentParser(description="Benchmark sector encodings.")
parser.add_argument("--data", default="data/facility_labeled.csv",
                    help="facility-level labeled CSV (default: v2 Alberta)")
args = parser.parse_args()

above = pd.read_csv(args.data).reset_index(drop=True)
print(f"data: {args.data}, facilities: {len(above)}, "
      f"sectors: {above['sector'].nunique()}")
above["log_emissions"] = np.log1p(above["avg_annual_emissions"])
y = above["priority"]
y_bin = (y == LABELS[0]).astype(int)

cv = StratifiedKFold(**CV_PARAMS)  # one object -> identical folds for all


def summarize(name, y_true, y_pred):
    r = classification_report(y_true, y_pred, labels=LABELS,
                              digits=4, output_dict=True)
    return {
        "encoding": name,
        "accuracy": round(r["accuracy"], 4),
        "macro_f1": round(r["macro avg"]["f1-score"], 4),
        "ccs_f1": round(r[LABELS[0]]["f1-score"], 4),
        "cu_f1": round(r[LABELS[1]]["f1-score"], 4),
        "cu_recall": round(r[LABELS[1]]["recall"], 4),
    }


results = []

# A: label baseline (v2 status quo)
X_label, _ = add_features(above)
pred = cross_val_predict(DecisionTreeClassifier(**MODEL_PARAMS),
                         X_label, y, cv=cv)
results.append(summarize("label (v2 status quo)", y, pred))

# B/C/D/E: target-family encoders inside pipelines (no leakage)
X_raw = above[["log_emissions", "sector", "years_reported"]].copy()
contenders = [
    ("target_s1", TargetEncoder(cols=["sector"], smoothing=1.0)),
    ("target_s10", TargetEncoder(cols=["sector"], smoothing=10.0)),
    ("target_s50", TargetEncoder(cols=["sector"], smoothing=50.0)),
    ("leave_one_out", LeaveOneOutEncoder(cols=["sector"])),
]
for name, enc in contenders:
    pipe = Pipeline([("enc", enc),
                     ("tree", DecisionTreeClassifier(**MODEL_PARAMS))])
    pred_bin = cross_val_predict(pipe, X_raw, y_bin, cv=cv)
    pred_labels = pd.Series(np.where(pred_bin == 1, LABELS[0], LABELS[1]),
                            index=y.index)
    results.append(summarize(name, y, pred_labels))

cmp_df = pd.DataFrame(results)
print("\n=== ENCODING BENCHMARK (identical 5-fold CV for all) ===")
print(cmp_df.to_string(index=False))

best = cmp_df.loc[cmp_df["macro_f1"].idxmax()]
base = cmp_df.iloc[0]
gap = best["macro_f1"] - base["macro_f1"]
print(f"\nbest: {best['encoding']} (macro F1 {best['macro_f1']})")
print(f"baseline: label (macro F1 {base['macro_f1']}); gap = {gap:+.4f}")
if best["encoding"] == "label (v2 status quo)":
    print("VERDICT: keep v2 label encoding. No contender beats it.")
elif gap < 0.03:
    print("VERDICT: best contender leads by < 0.03 (fold noise on n=150). "
          "Not worth the integration churn. Keep v2 label encoding; "
          "re-run on the national dataset where per-sector samples grow 10x.")
else:
    print(f"VERDICT: {best['encoding']} wins meaningfully. Ship it.")
