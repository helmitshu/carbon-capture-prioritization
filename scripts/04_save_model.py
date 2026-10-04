"""Phase 5a: train the final model and save deployment artifacts."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from features import add_features, FEATURES, LABELS, MODEL_PARAMS

above = pd.read_csv("data/facility_labeled.csv")

# features from the shared module: same derivation as 03, by construction
X, le = add_features(above)
y = above["priority"]

tree = DecisionTreeClassifier(**MODEL_PARAMS)
tree.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump({"model": tree, "encoder": le,
             "features": list(X.columns),
             "labels": LABELS},
            "model/model.pkl")
# facility lookup table for the app
above.to_csv("model/facilities.csv", index=False)
meta = {"n_facilities": len(above),
        "classes": y.value_counts().to_dict(),
        "trained": "2026-10-03"}
open("model/meta.json", "w").write(json.dumps(meta, indent=2))
print("saved model/model.pkl, model/facilities.csv, model/meta.json")
