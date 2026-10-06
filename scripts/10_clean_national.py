"""National cleaning (v3): staged refresh pipeline.

Stages, in order:
  1. Schema check  (scripts/schema_contract.py, hard stop on mismatch)
  2. Validation gate (scripts/validate.py, quarantine + 2% stop threshold)
  3. Cleaning v1    (frozen rule set, documented below)
  4. New v1 rules: year-over-year outlier flags (review list, never
     auto-dropped) and new/retired facility reconciliation.

Cleaning v1 is the frozen rule set inherited from the course-order
pipeline (02_clean), minus the Alberta filter:
  inspect, rename, province_audit, missing_values_first,
  duplicate_checks, type_coercion, valid_emissions_filter,
  temporal_check.
Missing-value policy: gas tonnes 0-filled where the CO2e counterpart is
present; company_trade forward-filled within facility then legal-name
fallback; city filled with "Unknown". Missing values are handled BEFORE
type coercion so coercion never silently creates fillable NaNs.

Reads the raw snapshot from data/raw/ (latest manifest release by
default). Writes data/Capstone_Dataset_clean_national.csv,
hidden_files/cleaning_report_national.json, and the data quality
report for the release. Raw files untouched. v2 files untouched.

Usage:
  python3 scripts/10_clean_national.py [--release-id 2024]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_contract import (SHORT_COLUMNS, check_schema,
                             check_schema_versioned, SCHEMAS)
from validate import run_validation_gate, EMISSION_COLS
from data_releases import (load_manifest, latest_release, release_path,
                           verify_release)
from quality_report import write_quality_report

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN = PROJECT_ROOT / "data" / "Capstone_Dataset_clean_national.csv"
REPORT_JSON = PROJECT_ROOT / "hidden_files" / "cleaning_report_national.json"
HIDDEN = PROJECT_ROOT / "hidden_files"

CLEANING_VERSION = "v1"
NUM_COLS = ["co2", "ch4_t", "ch4_co2e", "n2o_t", "n2o_co2e",
            "total_emissions", "year", "naics_code"]
CRITICAL_COLS = ["facility_id", "facility_name", "year", "naics_code",
                 "sector", "co2", "ch4_co2e", "n2o_co2e", "total_emissions"]
OUTLIER_PCT = 0.50  # YoY swing beyond this is flagged for review
OUTLIER_MIN_ABS_T = 10_000  # ...and the absolute swing must exceed this,
# so tiny facilities with noisy small bases do not flood the review list


def clean_v1(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Frozen cleaning rule set v1. Semantics must not change."""
    # 3a. Province audit (replaces the Alberta filter).
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

    # 3b. Missing values FIRST (same policy as v2).
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

    # 3c. Duplicate checks.
    exact_dupes = int(df.duplicated().sum())
    pair_dupes = int(df.duplicated(subset=["facility_id", "year"]).sum())
    report["exact_duplicate_rows"] = exact_dupes
    report["facility_year_duplicate_rows"] = pair_dupes
    if exact_dupes:
        df = df.drop_duplicates()
    assert pair_dupes == 0, "facility-year duplicates must be resolved"
    print(f"duplicates: exact={exact_dupes}, facility-year={pair_dupes}")

    # 3d. Type coercion, with an explicit report.
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

    # 3e. Valid-emissions filter.
    before = len(df)
    df = df[df["total_emissions"] > 0]
    report["zero_or_null_total_emissions_dropped"] = before - len(df)
    print(f"valid-emissions filter: {before} -> {len(df)}")

    # 3f. Temporal check.
    year_stats = df.groupby("year").agg(
        rows=("facility_id", "count"),
        missing_critical=("co2", lambda s: int(s.isna().sum())),
    )
    report["rows_by_year"] = {str(k): int(v) for k, v in
                              year_stats["rows"].items()}
    ymin, ymax = int(df["year"].min()), int(df["year"].max())
    report["temporal_decision"] = (
        f"Kept all years {ymin}-{ymax}, all provinces. Early years are "
        f"thinner due to reporting-threshold history but complete on "
        f"model-critical columns, so dropping them would destroy valid "
        f"data. Documented, not dropped."
    )
    report["rows_by_province_clean"] = {
        str(k): int(v) for k, v in df["province"].value_counts().items()
    }

    crit_missing = int(df[CRITICAL_COLS].isna().sum().sum())
    report["critical_columns_missing_after_cleaning"] = crit_missing
    assert crit_missing == 0, "critical columns still have missing values"
    return df


def flag_yoy_outliers(df: pd.DataFrame) -> list:
    """Flag facilities whose total emissions swung over OUTLIER_PCT
    against their own prior year, with an absolute swing above
    OUTLIER_MIN_ABS_T so tiny noisy bases do not flood the list.
    Review list only, never dropped."""
    work = df[["facility_id", "facility_name", "province", "year",
               "total_emissions"]].copy()
    work = work.sort_values(["facility_id", "year"])
    work["prev_emissions"] = work.groupby("facility_id")[
        "total_emissions"].shift(1)
    work["prev_year"] = work.groupby("facility_id")["year"].shift(1)
    valid = work["prev_emissions"].notna() & (work["prev_emissions"] > 0)
    work["pct_change"] = (
        (work["total_emissions"] - work["prev_emissions"])
        / work["prev_emissions"]).where(valid)
    work["abs_change"] = (
        work["total_emissions"] - work["prev_emissions"]).abs()
    flagged = work[(work["pct_change"].abs() > OUTLIER_PCT)
                   & (work["abs_change"] > OUTLIER_MIN_ABS_T)].copy()
    flagged = flagged.sort_values("abs_change", ascending=False)
    out = []
    for _, r in flagged.iterrows():
        out.append({
            "facility_id": str(r["facility_id"]),
            "facility_name": r["facility_name"],
            "province": r["province"],
            "year": int(r["year"]),
            "prev_year": int(r["prev_year"]),
            "prev_emissions": float(r["prev_emissions"]),
            "curr_emissions": float(r["total_emissions"]),
            "pct_change": round(float(r["pct_change"]), 4),
            "abs_change_tonnes": round(float(r["abs_change"]), 1),
        })
    return out


def reconcile_facilities(df: pd.DataFrame, release_year: int) -> dict:
    """New vs retired facilities against the prior reporting year."""
    first = df.groupby("facility_id")["year"].min()
    last = df.groupby("facility_id")["year"].max()
    info = df.drop_duplicates("facility_id").set_index("facility_id")[
        ["facility_name", "province"]]
    new_ids = first[first == release_year].index.tolist()
    retired_ids = last[last == release_year - 1].index.tolist()
    return {
        "facilities_in_release_year": int(df.loc[
            df["year"] == release_year, "facility_id"].nunique()),
        "new_facilities": [
            {"facility_id": str(i),
             "facility_name": info.loc[i, "facility_name"],
             "province": info.loc[i, "province"]}
            for i in sorted(new_ids)
        ],
        "retired_candidates": [
            {"facility_id": str(i),
             "facility_name": info.loc[i, "facility_name"],
             "province": info.loc[i, "province"],
             "last_year": int(last.loc[i])}
            for i in sorted(retired_ids)
        ],
    }


def main(release_id: str | None = None) -> dict:
    manifest = load_manifest()
    if release_id is None:
        release = latest_release()
        release_id = release["release_id"]
    else:
        matches = [r for r in manifest["releases"]
                   if r["release_id"] == release_id]
        if not matches:
            raise ValueError(f"unknown release_id: {release_id}")
        release = matches[0]
    if not verify_release(release):
        raise ValueError(
            f"Checksum mismatch on {release['file']}: the raw file on "
            f"disk does not match the manifest. Raw files are immutable, "
            f"investigate before proceeding.")
    raw_path = release_path(release)
    release_year = release["years_covered"][1]
    schema_version = release.get("schema_version", "v1")
    rename_map = SCHEMAS[schema_version]["rename"]
    print(f"release {release_id}: {raw_path.name} "
          f"(checksum ok, {release['rows']} rows)")

    report: dict = {
        "release_id": release_id,
        "cleaning_version": CLEANING_VERSION,
        "order_of_operations": [
            "schema_check", "validation_gate", "rename",
            "province_audit", "missing_values_first", "duplicate_checks",
            "type_coercion", "valid_emissions_filter", "temporal_check",
            "yoy_outlier_flags", "facility_reconciliation",
        ],
    }

    # Stage 1: schema check on the raw file.
    # utf-8-sig tolerates the BOM ECCC ships on newer releases.
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    report["raw_shape"] = list(raw.shape)
    report["raw_missing_by_column"] = {
        str(c): int(n) for c, n in raw.isna().sum().items() if n > 0
    }
    if schema_version == "v1":
        schema_report = check_schema(raw, release_id)
    else:
        schema_report = check_schema_versioned(raw, release_id,
                                               schema_version)
    report["schema_check"] = schema_report
    print(f"raw: {raw.shape}, schema {schema_version} ok "
          f"({schema_report['actual_columns']} columns)")

    # Stage 2: rename (v1 and v2 map to identical short names), then
    # the validation gate.
    df = raw.rename(columns=rename_map)
    df = df[SHORT_COLUMNS]
    assert list(df.columns) == SHORT_COLUMNS, "rename mismatch"
    report["raw_n_facilities"] = int(df["facility_id"].nunique())
    q_path = HIDDEN / f"quarantine_{release_id}.csv"
    df, val_report = run_validation_gate(df, release_year, q_path)
    report["validation_gate"] = val_report
    print(f"validation: {val_report['rows_clean']} clean, "
          f"{val_report['rows_quarantined']} quarantined "
          f"({val_report['quarantine_rate']:.2%})")

    # Stage 3: cleaning v1 (frozen).
    rows_before_clean = len(df)
    df = clean_v1(df, report)
    rows_out = len(df)
    report["cleaning_summary"] = {
        "rows_into_cleaning": rows_before_clean,
        "rows_out": rows_out,
        "rows_removed": rows_before_clean - rows_out,
    }

    # Stage 4: new v1 rules.
    outliers = flag_yoy_outliers(df)
    report["yoy_outlier_flags"] = {
        "threshold_pct": int(OUTLIER_PCT * 100),
        "min_abs_change_tonnes": OUTLIER_MIN_ABS_T,
        "flagged_count": len(outliers),
        "flagged": outliers,
    }
    print(f"YoY outliers flagged for review: {len(outliers)}")
    recon = reconcile_facilities(df, release_year)
    report["facility_reconciliation"] = recon
    print(f"new facilities: {len(recon['new_facilities'])}, "
          f"retired candidates: {len(recon['retired_candidates'])}")

    # Outputs.
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {CLEAN}: {df.shape}")
    print(f"wrote {REPORT_JSON}")

    qr = {
        "release_id": release_id,
        "source": release["source"],
        "retrieval_date": release["retrieval_date"],
        "years_covered": release["years_covered"],
        "cleaning_version": CLEANING_VERSION,
        "schema": schema_report,
        "validation": val_report,
        "cleaning": {
            "rows_out": rows_out,
            "rows_removed": rows_before_clean - rows_out,
            "removals": {
                "missing_province_rows_dropped":
                    report["missing_province_rows_dropped"],
                "gas_tonnes_unresolvable_dropped":
                    report["missing_value_policy"]
                    ["gas_tonnes_unresolvable_dropped"],
                "exact_duplicate_rows_dropped":
                    report["exact_duplicate_rows"],
                "zero_or_null_total_emissions_dropped":
                    report["zero_or_null_total_emissions_dropped"],
            },
        },
        "reconciliation": recon,
        "outliers": {"flagged": outliers,
                     "flagged_count": len(outliers)},
        "files": [
            str(CLEAN.relative_to(PROJECT_ROOT)),
            str(REPORT_JSON.relative_to(PROJECT_ROOT)),
            str(q_path.relative_to(PROJECT_ROOT)),
            str(raw_path.relative_to(PROJECT_ROOT)),
        ],
    }
    md_path, json_path = write_quality_report(qr, release_id, HIDDEN)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return qr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-id", default=None,
                    help="manifest release_id (default: latest)")
    args = ap.parse_args()
    main(args.release_id)
