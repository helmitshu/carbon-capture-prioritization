"""Phase 2+3: feature engineering + Decision Tree, faithful to the capstone lesson spec.
- Facility-level aggregation (mean across years)
- TIER filter: avg annual emissions >= 100,000 -> Above Threshold
- Labels: co2_share >= 0.85 -> CCS Candidate else Potential CU Candidate
- Features: log_emissions (log1p), naics_sector_encoded, years_reported
  (gas shares excluded: label leakage, per lesson)
- Model: DecisionTree(max_depth=3, min_samples_leaf=2, min_samples_split=6,
  class_weight=balanced), StratifiedKFold(5, shuffle, rs=42)
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/Capstone_Dataset_clean.csv")
# clean already emits lesson column names
print("rows:", len(df))

# facility-year duplicate check (lesson rule)
dupes = df.duplicated(subset=["facility_id", "year"]).sum()
print("facility-year duplicates:", dupes)
assert dupes == 0

# Step 1: aggregate to facility level
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
print("facilities:", len(fac))

# Step 2: gas shares
fac["co2_share"] = fac["avg_co2"] / fac["avg_annual_emissions"]
fac["ch4_share"] = fac["avg_ch4_co2e"] / fac["avg_annual_emissions"]
fac["n2o_share"] = fac["avg_n2o_co2e"] / fac["avg_annual_emissions"]

# Step 3: TIER threshold filter
TIER = 100_000
fac["emission_band"] = np.where(fac["avg_annual_emissions"] >= TIER,
                                "Above Threshold", "Below Threshold")
above = fac[fac["emission_band"] == "Above Threshold"].copy()
print(f"above-threshold facilities: {len(above)} of {len(fac)}")

# Step 4: labels (85% cutoff is arbitrary per lesson, not an industry standard)
CO2_THRESHOLD = 0.85
above["priority"] = above["co2_share"].apply(
    lambda x: "CCS Candidate" if x >= CO2_THRESHOLD else "Potential CU Candidate")
print(above["priority"].value_counts())

# Step 5: model features (gas shares excluded: leakage)
above["log_emissions"] = np.log1p(above["avg_annual_emissions"])
le = LabelEncoder()
above["naics_sector_encoded"] = le.fit_transform(above["sector"].astype(str))
X = above[["log_emissions", "naics_sector_encoded", "years_reported"]]
y = above["priority"]
print("feature dtypes ok:", X.dtypes.to_dict())

# Model + 5-fold stratified CV
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                              min_samples_split=6, class_weight="balanced",
                              random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(tree, X, y, cv=cv)
labels = ["CCS Candidate", "Potential CU Candidate"]
print("\n--- CLASSIFICATION REPORT (5-fold CV) ---")
print(classification_report(y, y_pred, labels=labels, digits=3))
print("--- CONFUSION MATRIX ---")
print(pd.DataFrame(confusion_matrix(y, y_pred, labels=labels),
                   index=["true CCS", "true CU"], columns=["pred CCS", "pred CU"]))

# Fit on full data for interpretation + visualization
tree.fit(X, y)
print("\n--- TREE (text) ---")
print(export_text(tree, feature_names=list(X.columns), max_depth=3))
print("\n--- FEATURE IMPORTANCES ---")
for f, imp in sorted(zip(X.columns, tree.feature_importances_),
                     key=lambda t: -t[1]):
    print(f"  {f}: {imp:.3f}")

plt.figure(figsize=(16, 9))
plot_tree(tree, feature_names=list(X.columns), class_names=labels,
          filled=True, rounded=True, fontsize=9)
plt.tight_layout()
plt.savefig("hidden_files/decision_tree.png", dpi=120)
print("\nsaved hidden_files/decision_tree.png")

# Step 6 (lesson): sector-level rollup
sec = above.groupby("sector", as_index=False).agg(
    facilities=("facility_id", "count"),
    avg_sector_emissions=("avg_annual_emissions", "mean"),
    ccs_candidates=("priority", lambda s: int((s == "CCS Candidate").sum())),
)
sec = sec.sort_values("ccs_candidates", ascending=False)
print("\n--- TOP SECTORS BY CCS CANDIDATES ---")
print(sec.head(8).to_string(index=False))

above.to_csv("data/facility_labeled.csv", index=False)
print("\nsaved data/facility_labeled.csv:", above.shape)
