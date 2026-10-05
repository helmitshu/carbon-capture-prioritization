"""National features + labels (v3): mirrors scripts/03_features_model.py
without the Alberta scope.

- Facility-level aggregation (mean across years)
- Neutral screening cutoff: avg annual emissions >= 100,000 tCO2e/yr
  (D2 default: the TIER threshold is Alberta-only, so nationally it is
  honestly labeled a large-emitter screening cutoff, not a regulation)
- Labels: co2_share >= 0.85 -> CCS Candidate else Potential CU Candidate
- Features: log_emissions, naics_sector_encoded (v2 label encoding;
  the 08 benchmark re-runs on this file to pick the v3 encoder),
  years_reported. Gas shares excluded: label leakage.

Writes data/facility_labeled_national.csv.
v2 files untouched.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from features import (add_features, make_labels, FEATURES, LABELS,
                      CO2_CUTOFF)

# D2 default (a): neutral national screening cutoff, honestly labeled.
NATIONAL_CUTOFF = 100_000  # tCO2e/yr, screening heuristic, not a regulation

df = pd.read_csv("data/Capstone_Dataset_clean_national.csv")
print("rows:", len(df))
assert int(df.duplicated(subset=["facility_id", "year"]).sum()) == 0

fac = df.groupby("facility_id", as_index=False).agg(
    facility_name=("facility_name", "first"),
    province=("province", "first"),
    city=("city", "first"),
    sector=("sector", "first"),
    naics_code=("naics_code", "first"),
    company_trade=("company_trade", "first"),
    avg_annual_emissions=("total_emissions", "mean"),
    avg_co2=("co2", "mean"),
    avg_ch4_co2e=("ch4_co2e", "mean"),
    avg_n2o_co2e=("n2o_co2e", "mean"),
    years_reported=("year", "nunique"),
)
print("facilities:", len(fac))

fac["co2_share"] = fac["avg_co2"] / fac["avg_annual_emissions"]
fac["ch4_share"] = fac["avg_ch4_co2e"] / fac["avg_annual_emissions"]
fac["n2o_share"] = fac["avg_n2o_co2e"] / fac["avg_annual_emissions"]

fac["emission_band"] = np.where(
    fac["avg_annual_emissions"] >= NATIONAL_CUTOFF,
    "Above Threshold", "Below Threshold")
above = fac[fac["emission_band"] == "Above Threshold"].copy()
print(f"above-cutoff facilities: {len(above)} of {len(fac)}")
print("by province:")
print(above["province"].value_counts().to_string())

above = make_labels(above)
print("\nlabels:")
print(above["priority"].value_counts().to_string())

X, _ = add_features(above)
above[["log_emissions", "naics_sector_encoded"]] = X[["log_emissions",
                                                     "naics_sector_encoded"]]

above.to_csv("data/facility_labeled_national.csv", index=False)
print(f"\nsaved data/facility_labeled_national.csv: {above.shape}")
print(f"sectors: {above['sector'].nunique()}")
