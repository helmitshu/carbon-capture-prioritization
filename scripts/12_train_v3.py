"""Train v3 (national) model: target-encoded sector, panel tiers, reasons.

- Features: log_emissions, naics_sector_encoded (target, smoothing=10,
  the 08-benchmark winner on national data), years_reported.
- Model: DecisionTreeClassifier with v2 MODEL_PARAMS (the judge stays
  explainable; the panel is the second opinion, not the verdict).
- Panel tiers: out-of-fold votes from 4 models (tree, random forest,
  logistic regression, grad boosting) -> panel_ccs_votes (0-4),
  verdict_tier (unanimous | majority | contested), panel_majority.
- Reasons: exact decision-path walk of the fitted tree per facility,
  rendered as plain-language sentences (no SHAP dependency; with 3
  features the tree path IS the explanation).

Writes:
  model/model_v3.pkl            bundle {model, encoder, features, labels}
  model/facilities_v3.csv       national facilities + tier/reason columns
                                (naics_sector_encoded is TARGET-encoded here)
  hidden_files/v3_metrics.json  CV metrics + limitations for the model card

v2 files untouched. data/facility_labeled_national.csv untouched
(its encoded column stays label-encoded; the v3 serving table carries
the target-encoded values).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

from features import (add_features, FEATURES, LABELS,
                      MODEL_PARAMS, CV_PARAMS)

NATIONAL_LABELED = "data/facility_labeled_national.csv"
SMOOTHING = 10.0  # 08-benchmark winner on national data


def describe_split(fname, went_left, thr, ctx):
    """One plain-language sentence for a tree split."""
    if fname == "log_emissions":
        thr_mt = (np.exp(thr) - 1) / 1e6
        return (f"Scale: {ctx['mt']:.2f} Mt/yr, "
                f"{'below' if went_left else 'above'} the "
                f"{thr_mt:.2f} Mt split")
    if fname == "naics_sector_encoded":
        return (f"Sector {ctx['sector']}: {ctx['sector_mean']:.0%} "
                f"candidate rate, {'below' if went_left else 'above'} "
                f"the {thr:.0%} split")
    if fname == "years_reported":
        return (f"Reporting history: {ctx['years']:.0f} years, "
                f"{'below' if went_left else 'above'} the "
                f"{thr:.0f}-year split")
    return f"{fname}: {'below' if went_left else 'above'} {thr:.3f}"


def facility_reasons(tree, x_row, ctx):
    """Reasons from the tree's exact decision path for one facility."""
    path = tree.decision_path(x_row.to_frame().T).indices[:-1]  # drop leaf
    out = []
    for node_id in path:
        f_idx = tree.tree_.feature[node_id]
        fname = FEATURES[f_idx]
        thr = tree.tree_.threshold[node_id]
        went_left = x_row[fname] <= thr
        out.append(describe_split(fname, went_left, thr, ctx))
    return out


def main():
    above = pd.read_csv(NATIONAL_LABELED).reset_index(drop=True)
    print(f"facilities: {len(above)}, sectors: {above['sector'].nunique()}")
    y = above["priority"]
    y_bin = (y == LABELS[0]).astype(int).to_numpy()

    # ---- v3 features: target-encoded sector ----
    X, adapter = add_features(above, encoding="target", y=y_bin,
                              smoothing=SMOOTHING)
    assert list(X.columns) == FEATURES

    # ---- the judge ----
    tree = DecisionTreeClassifier(**MODEL_PARAMS)
    tree.fit(X, y)
    print(f"tree nodes: {tree.tree_.node_count}, "
          f"depth: {tree.get_depth()}")

    # ---- the panel: out-of-fold votes, identical folds ----
    cv = StratifiedKFold(**CV_PARAMS)
    panel = {
        "tree": DecisionTreeClassifier(**MODEL_PARAMS),
        "rf": RandomForestClassifier(n_estimators=300,
                                     class_weight="balanced",
                                     random_state=42),
        "lr": Pipeline([("scale", StandardScaler()),
                        ("lr", LogisticRegression(
                            class_weight="balanced", max_iter=2000,
                            random_state=42))]),
        "gbm": HistGradientBoostingClassifier(class_weight="balanced",
                                              random_state=42),
    }
    votes = pd.DataFrame(index=above.index)
    for name, model in panel.items():
        pred = cross_val_predict(model, X, y, cv=cv)
        votes[name] = (pd.Series(pred, index=above.index)
                       == LABELS[0]).astype(int)
    n_votes = votes.sum(axis=1)
    above["panel_ccs_votes"] = n_votes.astype(int)
    above["panel_majority"] = np.where(n_votes >= 3, LABELS[0], LABELS[1])
    above["verdict_tier"] = np.where(
        n_votes.isin([0, 4]), "unanimous",
        np.where(n_votes.isin([1, 3]), "majority", "contested"))
    print("tiers:")
    print(above["verdict_tier"].value_counts().to_string())
    # tree-vs-majority agreement from the panel's own tree votes:
    tree_votes = votes["tree"]
    majority_ccs = (n_votes >= 3).astype(int)
    print(f"tree vs panel-majority agreement: "
          f"{(tree_votes == majority_ccs).mean():.1%}")

    # ---- reasons from the exact decision path ----
    means = dict(zip(adapter.categories_, adapter.values()))
    r1, r2, r3 = [], [], []
    for i, row in above.iterrows():
        ctx = {
            "mt": row["avg_annual_emissions"] / 1e6,
            "sector": row["sector"],
            "sector_mean": means[row["sector"]],
            "years": row["years_reported"],
        }
        rs = facility_reasons(tree, X.iloc[i], ctx)
        rs += ["", "", ""]
        r1.append(rs[0]); r2.append(rs[1]); r3.append(rs[2])
    above["reason_1"], above["reason_2"], above["reason_3"] = r1, r2, r3
    n_with = int((above["reason_1"] != "").sum())
    print(f"facilities with reasons: {n_with}/{len(above)}")
    assert n_with == len(above)

    # ---- target-encoded serving column ----
    above["naics_sector_encoded"] = X["naics_sector_encoded"].to_numpy()

    # ---- artifacts ----
    os.makedirs("model", exist_ok=True)
    joblib.dump({"model": tree, "encoder": adapter,
                 "features": list(X.columns), "labels": LABELS},
                "model/model_v3.pkl")
    above.to_csv("model/facilities_v3.csv", index=False)
    print("saved model/model_v3.pkl, model/facilities_v3.csv")

    # ---- CV metrics for the model card (leakage-safe protocol) ----
    from category_encoders import TargetEncoder
    pipe = Pipeline([
        ("te", TargetEncoder(cols=["sector"], smoothing=SMOOTHING,
                             handle_unknown="value",
                             handle_missing="value")),
        ("tree", DecisionTreeClassifier(**MODEL_PARAMS)),
    ])
    X_raw = above[["log_emissions", "sector",
                   "years_reported"]].copy()
    # NOTE: X_raw sector is the raw string here; above["sector"] is raw.
    X_raw["sector"] = pd.read_csv(
        NATIONAL_LABELED).reset_index(drop=True)["sector"]
    pred_bin = cross_val_predict(pipe, X_raw, y_bin, cv=cv)
    pred_labels = pd.Series(np.where(pred_bin == 1, LABELS[0], LABELS[1]),
                            index=y.index)
    r = classification_report(y, pred_labels, labels=LABELS,
                              digits=4, output_dict=True)
    metrics = {
        "protocol": (
            "TargetEncoder(smoothing=10, fit inside folds) + "
            "DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, "
            "min_samples_split=6, class_weight=balanced) with "
            "StratifiedKFold(5, shuffle, rs=42), cross_val_predict. "
            "No leakage: encoder never sees the validation fold."),
        "accuracy": round(r["accuracy"], 4),
        "macro_avg": {k: round(v, 4) for k, v in r["macro avg"].items()
                      if k in ("precision", "recall", "f1-score")},
        "per_class": {label: {
            "precision": round(r[label]["precision"], 4),
            "recall": round(r[label]["recall"], 4),
            "f1": round(r[label]["f1-score"], 4),
            "support": int(r[label]["support"])} for label in LABELS},
        "limitations": [
            "National screening heuristic: the 100kt cutoff is not a "
            "regulation outside Alberta; it is a large-emitter screen.",
            "CU class remains the weaker side; contested-tier facilities "
            "need human review, not silent verdicts.",
            "Target-encoded sector means are noisiest for sectors with "
            "few facilities; smoothing=10 shrinks them toward the mean.",
            "Single decision tree chosen for interpretability, not "
            "maximum performance (random forest scored +0.05 macro F1).",
        ],
    }
    with open("hidden_files/v3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"CV accuracy={metrics['accuracy']}, "
          f"macro F1={metrics['macro_avg']['f1-score']}")
    print("saved hidden_files/v3_metrics.json")


if __name__ == "__main__":
    main()
