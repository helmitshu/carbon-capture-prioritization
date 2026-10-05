"""Tests for the v2 course-order cleaning pipeline (scripts/02_clean.py).

Run: .venv/bin/python -m pytest tests/test_clean.py -q
(or .venv/bin/python tests/test_clean.py for a zero-dependency check)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "Capstone_Dataset_clean.csv"
REPORT = ROOT / "hidden_files" / "cleaning_report.json"

CRITICAL_COLS = ["facility_id", "facility_name", "year", "naics_code",
                 "sector", "co2", "ch4_co2e", "n2o_co2e", "total_emissions"]


def _load():
    return pd.read_csv(CLEAN)


def test_clean_file_exists():
    assert CLEAN.exists(), "cleaned CSV missing"


def test_report_exists_and_order():
    assert REPORT.exists(), "cleaning report missing"
    report = json.loads(REPORT.read_text())
    order = report["order_of_operations"]
    # missing values must come before type coercion (the course's headline rule)
    assert order.index("missing_values_first") < order.index("type_coercion")


def test_no_facility_year_duplicates():
    df = _load()
    assert df.duplicated(subset=["facility_id", "year"]).sum() == 0


def test_no_exact_duplicates():
    df = _load()
    assert df.duplicated().sum() == 0


def test_no_missing_in_critical_columns():
    df = _load()
    assert int(df[CRITICAL_COLS].isna().sum().sum()) == 0


def test_positive_total_emissions():
    df = _load()
    assert (df["total_emissions"] > 0).all()


def test_years_from_2004():
    df = _load()
    assert df["year"].min() >= 2004


def test_trade_names_mostly_repaired():
    df = _load()
    missing = df["company_trade"].isna().sum()
    # forward-fill + legal-name fallback should leave far fewer than raw
    assert missing < 3359, f"trade-name repair did not improve: {missing}"


def test_numeric_dtypes():
    df = _load()
    for c in ["co2", "ch4_co2e", "n2o_co2e", "total_emissions", "year"]:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} not numeric"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
