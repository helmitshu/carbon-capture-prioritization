"""Tests for the data refresh system: schema contract, validation
gate with quarantine, cleaning v1 stability, outlier flags.

Run: python3 -m pytest tests/test_data_system.py -q
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pandas as pd

from schema_contract import (CONTRACT, EXPECTED_COLUMNS, SHORT_COLUMNS,
                             check_schema, SchemaMismatch)
from validate import run_validation_gate, QuarantineGateError
from data_releases import QUARANTINE_STOP_RATE

ROOT = Path(__file__).resolve().parent.parent
RAW_SNAPSHOT = (ROOT / "data" / "raw"
                / "ghgrp_2004-2023_retrieved_2026-10-03.csv")
CLEAN_NATIONAL = ROOT / "data" / "Capstone_Dataset_clean_national.csv"


def _synthetic(n=100, **overrides):
    data = {
        "facility_id": [f"G{i:05d}" for i in range(n)],
        "year": [2023] * n,
        "facility_name": [f"Plant {i}" for i in range(n)],
        "city": ["Town"] * n,
        "province": ["Alberta"] * n,
        "npri_id": [float(i) for i in range(n)],
        "naics_code": [327310] * n,
        "sector": ["Cement"] * n,
        "sector_fr": ["Ciment"] * n,
        "company_legal": ["Corp"] * n,
        "company_trade": ["Trade"] * n,
        "co2": [100000.0] * n,
        "ch4_t": [10.0] * n,
        "ch4_co2e": [280.0] * n,
        "n2o_t": [1.0] * n,
        "n2o_co2e": [265.0] * n,
        "total_emissions": [100545.0] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)[SHORT_COLUMNS]


def _raw_frame():
    return pd.read_csv(RAW_SNAPSHOT, nrows=5)


# ---- schema contract ----

def test_schema_ok_on_real_raw_header():
    df = pd.read_csv(RAW_SNAPSHOT, nrows=5)
    report = check_schema(df, "2004-2023")
    assert report["schema_ok"] is True
    assert report["missing_columns"] == []
    assert report["extra_columns"] == []


def test_schema_missing_column_raises():
    df = _raw_frame().drop(columns=[EXPECTED_COLUMNS[0]])
    with pytest.raises(SchemaMismatch) as e:
        check_schema(df, "test")
    assert EXPECTED_COLUMNS[0] in str(e.value)


def test_schema_extra_column_raises():
    df = _raw_frame().copy()
    df["Surprise Column"] = 1
    with pytest.raises(SchemaMismatch) as e:
        check_schema(df, "test")
    assert "Surprise Column" in str(e.value)


def test_schema_renamed_column_raises():
    df = _raw_frame().rename(
        columns={EXPECTED_COLUMNS[11]: "CO2 (tonnes) RENAMED"})
    with pytest.raises(SchemaMismatch):
        check_schema(df, "test")


def test_schema_order_change_is_warning_only():
    df = _raw_frame()[list(reversed(EXPECTED_COLUMNS))]
    report = check_schema(df, "test")
    assert report["schema_ok"] is True
    assert report["order_changed"] is True


def test_contract_has_17_columns():
    assert len(CONTRACT) == 17
    assert len(EXPECTED_COLUMNS) == 17


# ---- validation gate ----

def test_gate_passes_clean_frame(tmp_path):
    df = _synthetic(100)
    q = tmp_path / "q.csv"
    clean, report = run_validation_gate(df, 2023, q)
    assert len(clean) == 100
    assert report["rows_quarantined"] == 0
    assert report["quarantine_rate"] == 0.0


def test_gate_quarantines_bad_rows_with_reasons(tmp_path):
    df = _synthetic(1000)
    df.loc[0, "facility_id"] = None            # null id
    df.loc[1, "year"] = 1999                   # out of range
    df.loc[2, "year"] = 2024                   # future vs release
    df.loc[3, "co2"] = "not-a-number"          # non numeric
    df.loc[4, "ch4_t"] = -5.0                  # negative
    df.loc[5, "facility_id"] = df.loc[6, "facility_id"]  # pair dupe
    q = tmp_path / "q.csv"
    clean, report = run_validation_gate(df, 2023, q)
    assert report["rows_quarantined"] == 6
    assert len(clean) == 994
    qdf = pd.read_csv(q)
    assert "quarantine_reason" in qdf.columns
    reasons = " ".join(qdf["quarantine_reason"].tolist())
    for needle in ("null_facility_id", "bad_year", "non_numeric_co2",
                   "negative_ch4_t", "facility_year_duplicate"):
        assert needle in reasons, needle


def test_gate_trips_over_two_percent(tmp_path):
    df = _synthetic(100)
    df.loc[:2, "facility_id"] = None  # 3% quarantined
    q = tmp_path / "q.csv"
    with pytest.raises(QuarantineGateError):
        run_validation_gate(df, 2023, q)
    # the quarantine file is still written for review
    assert q.exists()
    assert len(pd.read_csv(q)) == 3


def test_gate_exactly_at_threshold_passes(tmp_path):
    df = _synthetic(100)
    df.loc[:1, "facility_id"] = None  # exactly 2%
    q = tmp_path / "q.csv"
    clean, report = run_validation_gate(df, 2023, q)
    assert report["rows_quarantined"] == 2
    assert QUARANTINE_STOP_RATE == 0.02


# ---- cleaning v1 stability on the real baseline ----

def test_clean_v1_baseline_row_count():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    mod = importlib.import_module("10_clean_national")
    raw = pd.read_csv(RAW_SNAPSHOT).rename(
        columns={c["raw"]: c["short"] for c in CONTRACT})
    report: dict = {}
    out = mod.clean_v1(raw.copy(), report)
    assert len(out) == 18771
    assert report["missing_province_rows_dropped"] == 1
    assert report["facility_year_duplicate_rows"] == 0
    assert report["zero_or_null_total_emissions_dropped"] == 0
    assert report["critical_columns_missing_after_cleaning"] == 0


def test_clean_v1_golden_facility():
    df = pd.read_csv(CLEAN_NATIONAL)
    g = df[df["facility_id"] == "G10111"].sort_values("year")
    assert len(g) == 21  # 2004-2024 after the refresh
    assert int(g["year"].min()) == 2004
    assert int(g["year"].max()) == 2024
    v2023 = float(g[g["year"] == 2023]["total_emissions"].iloc[0])
    assert 100_000 < v2023 < 400_000
    assert bool((g["total_emissions"] > 0).all())


def test_yoy_outlier_flagging():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    mod = importlib.import_module("10_clean_national")
    df = pd.DataFrame({
        "facility_id": ["A", "A", "B", "B", "C", "C"],
        "facility_name": ["PA", "PA", "PB", "PB", "PC", "PC"],
        "province": ["Alberta"] * 6,
        "year": [2022, 2023, 2022, 2023, 2022, 2023],
        # A: big swing, B: small swing, C: big pct but tiny base
        "total_emissions": [100000.0, 200000.0, 100000.0, 140000.0,
                            100.0, 200.0],
    })
    flagged = mod.flag_yoy_outliers(df)
    assert len(flagged) == 1
    assert flagged[0]["facility_id"] == "A"
    assert flagged[0]["pct_change"] == pytest.approx(1.0)


def test_reconcile_facilities():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    mod = importlib.import_module("10_clean_national")
    df = pd.DataFrame({
        "facility_id": ["A", "A", "B", "C", "C"],
        "facility_name": ["PA", "PA", "PB", "PC", "PC"],
        "province": ["Alberta"] * 5,
        "year": [2022, 2023, 2023, 2022, 2022],
    })
    recon = mod.reconcile_facilities(df, 2023)
    assert [f["facility_id"] for f in recon["new_facilities"]] == ["B"]
    assert [f["facility_id"] for f in recon["retired_candidates"]] == ["C"]
    assert recon["facilities_in_release_year"] == 2


# ---- schema v2 (2024 release) ----

def test_schema_v2_ok_on_2024_raw():
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    import importlib as _il
    sc = _il.import_module("schema_contract")
    df = pd.read_csv(
        ROOT / "data" / "raw" / "ghgrp_2004-2024_retrieved_2026-10-06.csv",
        encoding="utf-8-sig", nrows=3)
    report = sc.check_schema_versioned(df, "2024", "v2")
    assert report["schema_ok"] is True
    assert report["expected_columns"] == 82
    # v2 maps to the identical 17 short names as v1
    assert sc.SHORT_COLUMNS_V2 == sc.SHORT_COLUMNS


def test_schema_v2_rejects_v1_file():
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    import importlib as _il
    sc = _il.import_module("schema_contract")
    df = pd.read_csv(RAW_SNAPSHOT, nrows=3)
    with pytest.raises(sc.SchemaMismatch):
        sc.check_schema_versioned(df, "2004-2023", "v2")
