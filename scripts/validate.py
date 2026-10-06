"""Validation gate for GHGRP raw data (renamed frame).

Runs AFTER the schema check and BEFORE cleaning v1. Every row that
fails a check is routed to a quarantine CSV with a reason column.
Nothing is silently dropped.

Checks:
  1. facility_id present (not null, not blank)
  2. year present and an integer within [2004, release_year]
  3. (facility_id, year) unique (first occurrence kept)
  4. numeric columns coerce to numbers
  5. emission columns are not negative

Locked decision: if the quarantine rate exceeds 2 percent, the refresh
stops with QuarantineGateError. A human reviews the quarantine file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_releases import QUARANTINE_STOP_RATE

EMISSION_COLS = ["co2", "ch4_t", "ch4_co2e", "n2o_t", "n2o_co2e",
                 "total_emissions"]
ID_COLS = ["facility_id", "year"]


class QuarantineGateError(Exception):
    """Raised when the quarantine rate exceeds the stop threshold."""


def run_validation_gate(df: pd.DataFrame, release_year: int,
                        quarantine_path: str | Path) -> dict:
    """Validate the renamed frame. Returns a counts report.

    Writes quarantined rows (with a quarantine_reason column) to
    quarantine_path. Returns (clean_df, report). Raises
    QuarantineGateError when the quarantine rate is over the threshold.
    """
    df = df.copy().reset_index(drop=True)
    n_in = len(df)
    reasons = pd.Series([""] * n_in, index=df.index, dtype=object)

    def flag(mask, reason):
        nonlocal reasons
        m = mask.fillna(False).astype(bool)
        hit = m & (reasons == "")
        reasons = reasons.mask(hit, reason)
        both = m & (reasons != "") & (reasons != reason)
        reasons = reasons.mask(both, reasons[both] + ";" + reason)

    # 1. facility_id present.
    fid = df["facility_id"]
    flag(fid.isna() | (fid.astype(str).str.strip() == ""),
         "null_facility_id")

    # 2. year present and in range.
    yr = pd.to_numeric(df["year"], errors="coerce")
    flag(yr.isna() | (yr % 1 != 0) | (yr < 2004) | (yr > release_year),
         "bad_year")

    # 3. (facility_id, year) unique, first kept.
    flag(df.duplicated(subset=ID_COLS, keep="first"),
         "facility_year_duplicate")

    # 4 and 5. numeric coercion and non-negativity per emission column.
    for col in EMISSION_COLS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        flag(coerced.isna() & df[col].notna(), f"non_numeric_{col}")
        flag(coerced < 0, f"negative_{col}")

    quarantined = reasons != ""
    q_df = df.loc[quarantined].copy()
    q_df["quarantine_reason"] = reasons[quarantined]
    clean_df = df.loc[~quarantined].copy().reset_index(drop=True)

    n_q = int(quarantined.sum())
    rate = n_q / n_in if n_in else 0.0
    report = {
        "rows_in": n_in,
        "rows_quarantined": n_q,
        "rows_clean": len(clean_df),
        "quarantine_rate": round(rate, 6),
        "quarantine_stop_threshold": QUARANTINE_STOP_RATE,
        "quarantine_reasons": {
            str(k): int(v)
            for k, v in q_df["quarantine_reason"].value_counts().items()
        } if n_q else {},
        "quarantine_file": str(quarantine_path),
    }

    quarantine_path = Path(quarantine_path)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(quarantine_path, index=False)

    if rate > QUARANTINE_STOP_RATE:
        raise QuarantineGateError(
            f"Quarantine rate {rate:.2%} exceeds the "
            f"{QUARANTINE_STOP_RATE:.0%} stop threshold "
            f"({n_q} of {n_in} rows). Review {quarantine_path}."
        )
    return clean_df, report
