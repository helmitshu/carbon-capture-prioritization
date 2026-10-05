"""National cleaning (v3): same course-order pipeline as scripts/02_clean.py,
minus the Alberta filter.

Reuses RENAME / NUM_COLS / CRITICAL_COLS from 02_clean so the schema stays
identical; only the province scope changes. One row has a missing
province and is dropped (unattributable), reported explicitly.

Writes data/Capstone_Dataset_clean_national.csv and
hidden_files/cleaning_report_national.json. Raw file untouched.
v2 files (Capstone_Dataset_clean.csv, cleaning_report.json) untouched.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_clean2 = importlib.import_module("02_clean")
RENAME = _clean2.RENAME
NUM_COLS = _clean2.NUM_COLS
CRITICAL_COLS = _clean2.CRITICAL_COLS

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "Capstone_Dataset.csv"
CLEAN = PROJECT_ROOT / "data" / "Capstone_Dataset_clean_national.csv"
REPORT = PROJECT_ROOT / "hidden_files" / "cleaning_report_national.json"


def main() -> None:
    report: dict = {"order_of_operations": [
        "inspect", "rename", "province_audit", "missing_values_first",
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

    # 2. Rename (identical schema to v2).
    df = df.rename(columns=RENAME)
    assert list(df.columns) == list(RENAME.values()), "rename mismatch"

    # 3. Province audit (replaces the Alberta filter).
    df["province"] = df["province"].str.strip()
    prov_counts = df["province"].value_counts(dropna=False)
    report["rows_by_province_raw"] = {
        str(k): int(v) for k, v in prov_counts.items()
    }
    missing_prov = int(df["province"].isna().sum())
    report["missing_province_rows_dropped"] = missing_prov
    if missing_prov:
        df = df[df["province"].notna()].copy()
    print(f"provinces: {df['province'].nunique()}, "
          f"dropped {missing_prov} unattributable rows")

    # 4. Missing values FIRST (same policy as v2).
    mv: dict = {}
    for tonnes, co2e in [("ch4_t", "ch4_co2e"), ("n2o_t", "n2o_co2e")]:
        mask = df[tonnes].isna() & df[co2e].notna()
        mv[f"{tonnes}_zero_filled_with_{co2e}_present"] = int(mask.sum())
        df.loc[mask, tonnes] = 0.0
    left = int(df[["ch4_t", "n2o_t"]].isna().sum().sum())
    mv["gas_tonnes_unresolvable_dropped"] = 0
    if left:
        before = len(df)
        df = df.dropna(subset=["ch4_t", "n2o_t"])
        mv["gas_tonnes_unresolvable_dropped"] = before - len(df)
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
    mv["city_filled_unknown"] = int(df["city"].isna().sum())
    df["city"] = df["city"].fillna("Unknown")
    report["missing_value_policy"] = mv

    # 5. Duplicate checks.
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

    # 7. Valid-emissions filter.
    before = len(df)
    df = df[df["total_emissions"] > 0]
    report["zero_or_null_total_emissions_dropped"] = before - len(df)
    print(f"valid-emissions filter: {before} -> {len(df)}")

    # 8. Temporal check.
    year_stats = df.groupby("year").agg(
        rows=("facility_id", "count"),
        missing_critical=("co2", lambda s: int(s.isna().sum())),
    )
    report["rows_by_year"] = {str(k): int(v) for k, v in
                              year_stats["rows"].items()}
    report["temporal_decision"] = (
        "Kept all years 2004-2023, all provinces. Same rationale as v2: "
        "early years are thinner due to reporting-threshold history but "
        "complete on model-critical columns."
    )
    report["rows_by_province_clean"] = {
        str(k): int(v) for k, v in df["province"].value_counts().items()
    }

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
