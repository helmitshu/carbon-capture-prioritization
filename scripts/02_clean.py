"""Phase 1 cleaning (v2): course-order data cleaning.

Order of operations follows the capstone lesson's data walkthrough:
  1. Inspect (shape, per-column missingness) before touching anything.
  2. Rename bilingual headers to short names.
  3. Alberta filter.
  4. Missing values FIRST, with an explicit per-column policy, BEFORE
     type coercion (so coercion never silently creates fillable NaNs):
       - gas tonnes (ch4_t, n2o_t): 0-fill where the CO2e counterpart is
         present (evidence the facility reported that year);
       - company_trade: forward-fill within facility_id ordered by year,
         then fall back to company_legal;
       - city: "Unknown" (display-only column).
  5. Duplicate checks: exact duplicate rows AND facility-year pairs.
  6. Type coercion to numeric, with a report of coerced values.
  7. Valid-emissions filter: drop zero/null total_emissions.
  8. Temporal check: verify completeness from 2004 onward. Early years
     (2004-2008) are thinner due to reporting-threshold history, but their
     rows are complete on model-critical columns, so they are kept with
     this rationale documented (dropping complete rows would destroy data).

Writes data/Capstone_Dataset_clean.csv and
hidden_files/cleaning_report.json. Raw file untouched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "Capstone_Dataset.csv"
CLEAN = PROJECT_ROOT / "data" / "Capstone_Dataset_clean.csv"
REPORT = PROJECT_ROOT / "hidden_files" / "cleaning_report.json"

RENAME = {
    "GHGRP ID No. / No d'identification du PDGES": "facility_id",
    "Reference Year / Année de référence": "year",
    "Facility Name / Nom de l'installation": "facility_name",
    "Facility City or District or Municipality / Ville ou District ou Municipalité de l'installation": "city",
    "Facility Province or Territory / Province ou territoire de l'installation": "province",
    "Facility NPRI ID / Numéro d'identification de l'INRP": "npri_id",
    "Facility NAICS Code / Code SCIAN de l'installation": "naics_code",
    "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais": "sector",
    "French Facility NAICS Code Description / Description du code SCIAN de l'installation en français": "sector_fr",
    "Reporting Company Legal Name / Dénomination sociale de la société déclarante": "company_legal",
    "Reporting Company Trade Name / Nom commercial de la société déclarante": "company_trade",
    "CO2 (tonnes)": "co2",
    "CH4 (tonnes)": "ch4_t",
    "CH4 (tonnes CO2e / tonnes éq. CO2)": "ch4_co2e",
    "N2O (tonnes)": "n2o_t",
    "N2O (tonnes CO2e / tonnes éq. CO2)": "n2o_co2e",
    "Total Emissions (tonnes CO2e) / Émissions totales (tonnes éq. CO2)": "total_emissions",
}

NUM_COLS = ["co2", "ch4_t", "ch4_co2e", "n2o_t", "n2o_co2e",
            "total_emissions", "year", "naics_code"]
CRITICAL_COLS = ["facility_id", "facility_name", "year", "naics_code",
                 "sector", "co2", "ch4_co2e", "n2o_co2e", "total_emissions"]


def main() -> None:
    report: dict = {"order_of_operations": [
        "inspect", "rename", "alberta_filter", "missing_values_first",
        "duplicate_checks", "type_coercion", "valid_emissions_filter",
        "temporal_check",
    ]}
    df = pd.read_csv(RAW)

    # 1. Inspect before touching anything.
    report["raw_shape"] = list(df.shape)
    report["raw_missing_by_column"] = {
        c: int(n) for c, n in df.isna().sum().items() if n > 0
    }
    print(f"raw: {df.shape}")

    # 2. Rename.
    df = df.rename(columns=RENAME)
    assert list(df.columns) == list(RENAME.values()), "rename mismatch"

    # 3. Alberta filter.
    df = df[df["province"].str.contains("Alberta", na=False)].copy()
    report["alberta_rows"] = len(df)
    print(f"Alberta rows: {len(df)}")

    # 4. Missing values FIRST (before type coercion).
    mv: dict = {}
    # 4a. Gas tonnes: 0-fill only where the CO2e counterpart is present.
    for tonnes, co2e in [("ch4_t", "ch4_co2e"), ("n2o_t", "n2o_co2e")]:
        mask = df[tonnes].isna() & df[co2e].notna()
        mv[f"{tonnes}_zero_filled_with_{co2e}_present"] = int(mask.sum())
        df.loc[mask, tonnes] = 0.0
    # Any gas-tonne NaN left without a CO2e counterpart is unresolvable.
    left = int(df[["ch4_t", "n2o_t"]].isna().sum().sum())
    mv["gas_tonnes_unresolvable_dropped"] = 0
    if left:
        before = len(df)
        df = df.dropna(subset=["ch4_t", "n2o_t"])
        mv["gas_tonnes_unresolvable_dropped"] = before - len(df)
    # 4b. Trade names: forward-fill within facility, then legal-name fallback.
    df = df.sort_values(["facility_id", "year"]).reset_index(drop=True)
    tn_before = int(df["company_trade"].isna().sum())
    df["company_trade"] = df.groupby("facility_id")["company_trade"].ffill()
    tn_after_ffill = int(df["company_trade"].isna().sum())
    df["company_trade"] = df["company_trade"].fillna(df["company_legal"])
    tn_after_legal = int(df["company_trade"].isna().sum())
    mv["trade_name_missing_before"] = tn_before
    mv["trade_name_filled_by_ffill"] = tn_before - tn_after_ffill
    mv["trade_name_filled_by_legal_name"] = tn_after_ffill - tn_after_legal
    mv["trade_name_still_missing"] = tn_after_legal
    # 4c. City is display-only.
    mv["city_filled_unknown"] = int(df["city"].isna().sum())
    df["city"] = df["city"].fillna("Unknown")
    report["missing_value_policy"] = mv
    print(f"trade names: {tn_before} missing -> {tn_after_legal} still missing")

    # 5. Duplicate checks (both kinds the lesson requires).
    exact_dupes = int(df.duplicated().sum())
    pair_dupes = int(df.duplicated(subset=["facility_id", "year"]).sum())
    report["exact_duplicate_rows"] = exact_dupes
    report["facility_year_duplicate_rows"] = pair_dupes
    if exact_dupes:
        df = df.drop_duplicates()
    assert pair_dupes == 0, "facility-year duplicates must be resolved"
    print(f"duplicates: exact={exact_dupes}, facility-year={pair_dupes}")

    # 6. Type coercion, with an explicit report.
    coerced: dict = {}
    for c in NUM_COLS:
        bad = df[c][pd.to_numeric(df[c], errors="coerce").isna()
                    & df[c].notna()]
        coerced[c] = {
            "coerced_values": int(len(bad)),
            "examples": sorted(bad.astype(str).unique().tolist())[:5],
        }
        df[c] = pd.to_numeric(df[c], errors="coerce")
    report["type_coercion"] = coerced

    # 7. Valid-emissions filter: drop zero/null totals.
    before = len(df)
    df = df[df["total_emissions"] > 0]
    report["zero_or_null_total_emissions_dropped"] = before - len(df)
    print(f"valid-emissions filter: {before} -> {len(df)}")

    # 8. Temporal check: completeness from 2004 onward; keep all years.
    year_stats = df.groupby("year").agg(
        rows=("facility_id", "count"),
        missing_critical=("co2", lambda s: int(s.isna().sum())),
    )
    report["rows_by_year"] = {str(k): int(v) for k, v in
                              year_stats["rows"].items()}
    report["temporal_decision"] = (
        "Kept all years 2004-2023. Early years (2004-2008) are thinner "
        "(~100 rows/yr vs ~700+/yr from 2017) due to reporting-threshold "
        "history, but their rows are complete on model-critical columns, "
        "so dropping them would destroy valid data. Documented, not dropped."
    )

    # Final invariant: no missing values in model-critical columns.
    crit_missing = int(df[CRITICAL_COLS].isna().sum().sum())
    report["critical_columns_missing_after_cleaning"] = crit_missing
    assert crit_missing == 0, "critical columns still have missing values"

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {CLEAN}: {df.shape}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
