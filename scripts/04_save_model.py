"""Phase 5a: train the final model and save deployment artifacts."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

above = pd.read_csv("data/facility_labeled.csv")

# re-derive features exactly as in 03 (kept in sync by construction)
above["log_emissions"] = np.log1p(above["avg_annual_emissions"])
le = LabelEncoder()
above["naics_sector_encoded"] = le.fit_transform(above["sector"].astype(str))
X = above[["log_emissions", "naics_sector_encoded", "years_reported"]]
y = above["priority"]

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                              min_samples_split=6, class_weight="balanced",
                              random_state=42)
tree.fit(X, y)

import os
os.makedirs("model", exist_ok=True)
joblib.dump({"model": tree, "encoder": le,
             "features": list(X.columns),
             "labels": ["CCS Candidate", "Potential CU Candidate"]},
            "model/model.pkl")
# facility lookup table for the app
above.to_csv("model/facilities.csv", index=False)
meta = {"n_facilities": len(above),
        "classes": y.value_counts().to_dict(),
        "trained": "2026-10-03"}
open("model/meta.json", "w").write(json.dumps(meta, indent=2))
print("saved model/model.pkl, model/facilities.csv, model/meta.json")
