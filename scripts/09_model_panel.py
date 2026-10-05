"""Panel of judges: does the v2 decision tree agree with other math models?

Four classifiers with genuinely different reasoning, identical 5-fold CV
(same StratifiedKFold object, same row order, same features), out-of-fold
predictions for every facility. Two questions:
  1. Do the models score similarly? (If the tree is far worse, rethink it.)
  2. Do they flag the SAME facilities? (Agreement = the verdicts hold up.)

Usage: python3 scripts/09_model_panel.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

from features import (add_features, make_labels, LABELS,
                      MODEL_PARAMS, CV_PARAMS, TIER_THRESHOLD)

# ---- same data prep as scripts/03 and 08 (canonical row order) ----
df = pd.read_csv("data/Capstone_Dataset_clean.csv")
fac = df.groupby("facility_id", as_index=False).agg(
    facility_name=("facility_name", "first"),
    sector=("sector", "first"),
    naics_code=("naics_code", "first"),
    avg_annual_emissions=("total_emissions", "mean"),
    avg_co2=("co2", "mean"),
    avg_ch4_co2e=("ch4_co2e", "mean"),
    avg_n2o_co2e=("n2o_co2e", "mean"),
    years_reported=("year", "nunique"),
)
fac["co2_share"] = fac["avg_co2"] / fac["avg_annual_emissions"]
above = fac[fac["avg_annual_emissions"] >= TIER_THRESHOLD].copy()
above = make_labels(above)
above = above.reset_index(drop=True)

X, _ = add_features(above)   # v2 features, label-encoded sector
y = above["priority"]
cv = StratifiedKFold(**CV_PARAMS)

panel = {
    "decision_tree (v2)": DecisionTreeClassifier(**MODEL_PARAMS),
    "random_forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42),
    "logistic_regression": Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced",
                                  max_iter=2000, random_state=42))]),
    "grad_boosting": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42),
}

oof = {}   # out-of-fold predictions per model
rows = []
for name, model in panel.items():
    pred = cross_val_predict(model, X, y, cv=cv)
    oof[name] = pd.Series(pred, index=above.index)
    r = classification_report(y, pred, labels=LABELS, output_dict=True)
    rows.append({
        "model": name,
        "accuracy": round(r["accuracy"], 4),
        "macro_f1": round(r["macro avg"]["f1-score"], 4),
        "ccs_f1": round(r[LABELS[0]]["f1-score"], 4),
        "cu_f1": round(r[LABELS[1]]["f1-score"], 4),
        "cu_recall": round(r[LABELS[1]]["recall"], 4),
    })

print("\n=== PANEL SCORES (identical 5-fold CV) ===")
print(pd.DataFrame(rows).to_string(index=False))

# ---- agreement: for each facility, how many of the 4 say CCS? ----
votes = pd.DataFrame(
    {n: (s == LABELS[0]).astype(int) for n, s in oof.items()})
votes["n_ccs_votes"] = votes.sum(axis=1)
dist = votes["n_ccs_votes"].value_counts().sort_index()
print("\n=== AGREEMENT (votes for CCS Candidate per facility) ===")
for v in range(5):
    n = int(dist.get(v, 0))
    label = ("unanimous CU" if v == 0 else "unanimous CCS" if v == 4
             else f"{v}/4 for CCS")
    print(f"  {label}: {n} facilities")

tree_pred = oof["decision_tree (v2)"]
majority = votes["n_ccs_votes"] >= 3
tree_says_ccs = tree_pred == LABELS[0]
disagree = above[tree_says_ccs != majority][
    ["facility_name", "sector", "avg_annual_emissions", "co2_share"]]
print(f"\nfacilities where the tree disagrees with the panel majority: "
      f"{len(disagree)} of {len(above)}")
if len(disagree):
    disagree = disagree.copy()
    disagree["tree_says"] = tree_pred[disagree.index].values
    disagree["panel_majority"] = np.where(
        majority[disagree.index], LABELS[0], LABELS[1])
    print(disagree.to_string(index=False))

agree_rate = (tree_says_ccs == majority).mean()
print(f"\ntree vs panel-majority agreement: {agree_rate:.1%}")
if agree_rate >= 0.90:
    print("VERDICT: the tree's verdicts hold up. The panel concurs.")
else:
    print("VERDICT: material disagreement. Review the listed facilities "
          "before trusting the tree alone.")
